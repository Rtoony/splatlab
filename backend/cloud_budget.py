"""Durable, aggregate admission accounting for explicitly authorized cloud work."""

from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time

MAX_CAP_MICROUSD = 50_000_000
REQUEST_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")


class BudgetRefused(ValueError):
    pass


class CloudBudget:
    def __init__(self, path):
        self.path = Path(path).absolute()
        if self.path.resolve() != self.path:
            raise BudgetRefused("Cloud ledger paths must not be symlinked")

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, cap_microusd, deadline_utc):
        if type(cap_microusd) is not int or not 0 < cap_microusd <= MAX_CAP_MICROUSD:
            raise BudgetRefused("Explicit cloud cap must be positive and at most $50")
        if type(deadline_utc) is not int or deadline_utc <= int(time.time()):
            raise BudgetRefused("An unexpired, explicit UTC deadline is required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("xb"):
                pass
            self.path.chmod(0o600)
        except FileExistsError:
            pass
        with self.connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS authorization (singleton INTEGER PRIMARY KEY CHECK(singleton=1), cap_microusd INTEGER NOT NULL, deadline_utc INTEGER NOT NULL, created_utc INTEGER NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, reserved_microusd INTEGER NOT NULL CHECK(reserved_microusd>0), request_sha256 TEXT NOT NULL, limits_json TEXT NOT NULL, created_utc INTEGER NOT NULL, status TEXT NOT NULL, result_json TEXT)")
            existing = connection.execute("SELECT * FROM authorization WHERE singleton=1").fetchone()
            if existing:
                if (existing["cap_microusd"], existing["deadline_utc"]) != (cap_microusd, deadline_utc):
                    raise BudgetRefused("Existing authorization cannot be raised, extended, or reset")
            else:
                connection.execute("INSERT INTO authorization VALUES (1,?,?,?)", (cap_microusd, deadline_utc, int(time.time())))
        return self.status()

    @staticmethod
    def authorization(connection):
        record = connection.execute("SELECT * FROM authorization WHERE singleton=1").fetchone()
        if not record or not 0 < record["cap_microusd"] <= MAX_CAP_MICROUSD:
            raise BudgetRefused("Missing or invalid explicit cloud authorization")
        return record

    def require_open(self, duration_seconds=240):
        with self.connection() as connection:
            record = self.authorization(connection)
            if time.time() + duration_seconds >= record["deadline_utc"]:
                raise BudgetRefused("Cloud window expired or too short for a bounded request")

    def reserve(self, request_id, provider, model, amount_microusd, request_sha256, limits):
        if not REQUEST_RE.fullmatch(request_id) or not re.fullmatch(r"[a-f0-9]{64}", request_sha256):
            raise BudgetRefused("Invalid request identity")
        if type(amount_microusd) is not int or amount_microusd <= 0:
            raise BudgetRefused("Reservation must be a positive integer number of microdollars")
        with self.connection() as connection:
            record = self.authorization(connection)
            if time.time() + 240 >= record["deadline_utc"]:
                raise BudgetRefused("Cloud window expired or too short for a bounded request")
            if connection.execute("SELECT 1 FROM requests WHERE request_id=?", (request_id,)).fetchone():
                raise BudgetRefused("Request already admitted; never retry an uncertain paid request")
            held = connection.execute("SELECT COALESCE(SUM(reserved_microusd),0) FROM requests").fetchone()[0]
            if held + amount_microusd > record["cap_microusd"]:
                raise BudgetRefused("Aggregate cloud spending cap would be exceeded")
            connection.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,NULL)", (
                request_id, provider, model, amount_microusd, request_sha256,
                json.dumps(limits, sort_keys=True), int(time.time()), "reserved",
            ))

    def finish(self, request_id, status, result):
        if status not in {"returned", "failed-unknown-charge"}:
            raise BudgetRefused("Unknown cloud outcome")
        with self.connection() as connection:
            changed = connection.execute("UPDATE requests SET status=?,result_json=? WHERE request_id=? AND status='reserved'", (
                status, json.dumps(result, sort_keys=True), request_id,
            ))
            if changed.rowcount != 1:
                raise BudgetRefused("Only an outstanding reservation can be finalized")

    def status(self):
        with self.connection() as connection:
            record = dict(self.authorization(connection))
            requests = [dict(row) for row in connection.execute("SELECT * FROM requests ORDER BY created_utc,request_id")]
        for request in requests:
            request["limits"] = json.loads(request.pop("limits_json"))
            request["result"] = json.loads(request.pop("result_json") or "null")
        held = sum(request["reserved_microusd"] for request in requests)
        return {"cap_microusd": record["cap_microusd"], "deadline_utc": record["deadline_utc"],
                "held_microusd": held, "available_microusd": record["cap_microusd"] - held,
                "expired": time.time() >= record["deadline_utc"], "requests": requests,
                "accounting": "Conservative reservations never released, including failures; not a provider invoice"}

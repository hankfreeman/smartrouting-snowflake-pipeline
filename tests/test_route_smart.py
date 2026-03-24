#!/usr/bin/env python3
"""
Test script for the Smart Routing /route endpoint (unified: TransUnion + preprocessing + 5-tower).

Usage:
  # Snowflake SPCS — set BASE_URL to your service ingress first
  set BASE_URL=https://<your-subdomain>.snowflakecomputing.app
  python test_route_smart.py
  python test_route_smart.py 5555550100 "Example Lead Source"

  # Override endpoint for one run
  set BASE_URL=https://<your-subdomain>.snowflakecomputing.app
  python test_route_smart.py 5555550100 "Example Lead"

  # With OAuth: set PRIVATE_KEY_PATH (e.g. rsa_key.p8), or set SNOWFLAKE_TOKEN.

Environment:
  BASE_URL          Default: new 5-tower Snowflake endpoint.
  SNOWFLAKE_TOKEN   Optional. If set and BASE_URL is https, use as bearer token.
  PRIVATE_KEY_PATH  Path to RSA key for OAuth (e.g. rsa_key.p8).
"""
import json
import os
import sys
import time

import requests

# -----------------------------------------------------------------------------
# Config: set BASE_URL and SF_* (for OAuth) via environment — no secrets in repo
# -----------------------------------------------------------------------------
BASE_URL = (os.environ.get("BASE_URL") or "").rstrip("/")
SF_ACCOUNT_URL = os.environ.get("SF_ACCOUNT_URL", "")
SF_ACCOUNT_LOCATOR = os.environ.get("SF_ACCOUNT_LOCATOR", "")
SF_USER_NAME = os.environ.get("SF_USER_NAME", "")
DEFAULT_PHONE = "5555550100"
DEFAULT_LEAD_SOURCE = "Example Lead"
TIMEOUT = 60


def _spcs_scope_from_base_url(url: str) -> str:
    if not url.startswith("https://"):
        return ""
    return url.replace("https://", "").split("/")[0].strip().lower()


def get_snowflake_token():
    """Get SPCS token via JWT bearer (RSA key + OAuth). Requires PyJWT and cryptography."""
    key_path = os.environ.get("PRIVATE_KEY_PATH", "rsa_key.p8")
    if not os.path.isfile(key_path):
        return None
    try:
        import hashlib
        import base64
        import jwt
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        print("(OAuth requires: pip install PyJWT cryptography)", file=sys.stderr)
        return None
    account_url = SF_ACCOUNT_URL
    account_locator = SF_ACCOUNT_LOCATOR
    user_name = SF_USER_NAME
    scope = os.environ.get("SPCS_ENDPOINT_HOSTNAME", "") or _spcs_scope_from_base_url(BASE_URL)
    if not all([account_url, account_locator, user_name, scope]):
        return None
    try:
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        public_key_bytes = private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        fingerprint = "SHA256:" + base64.b64encode(
            hashlib.sha256(public_key_bytes).digest()
        ).decode()
        qualified_user = f"{account_locator}.{user_name}"
        now = int(time.time())
        signed_jwt = jwt.encode(
            {
                "iss": f"{qualified_user}.{fingerprint}",
                "sub": qualified_user,
                "iat": now,
                "exp": now + 3600,
            },
            key=private_key,
            algorithm="RS256",
        )
        resp = requests.post(
            f"https://{account_url}.snowflakecomputing.com/oauth/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "scope": scope,
                "assertion": signed_jwt,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        return resp.text.strip()
    except Exception:
        return None


def call_route(phone_number: str, lead_source: str) -> dict:
    """POST /route and return JSON."""
    url = f"{BASE_URL}/route"
    payload = {"phone_number": phone_number, "lead_source": lead_source}
    headers = {"Content-Type": "application/json"}

    if BASE_URL.startswith("https://"):
        token = os.environ.get("SNOWFLAKE_TOKEN") or get_snowflake_token()
        if token:
            headers["Authorization"] = f'Snowflake Token="{token}"'
        elif not os.environ.get("SNOWFLAKE_TOKEN"):
            print("(Snowflake: ensure rsa_key.p8 exists and PRIVATE_KEY_PATH is set if needed.)", file=sys.stderr)

    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    try:
        data = resp.json()
    except Exception:
        data = {"success": False, "error": resp.text[:500]}
    if not resp.ok:
        data["_status_code"] = resp.status_code
        if resp.text.strip().lower().startswith("<!doctype html>") or "<html" in resp.text[:200].lower():
            data["_auth_hint"] = "Endpoint returned login page. Use RSA key OAuth: set PRIVATE_KEY_PATH to your .p8 key."
    return data


def main():
    phone = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PHONE
    lead_source = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LEAD_SOURCE
    show_full_json = "--json" in sys.argv or "-j" in sys.argv

    if not BASE_URL:
        print("Set BASE_URL to your SPCS HTTPS endpoint, e.g. set BASE_URL=https://<subdomain>.snowflakecomputing.app", file=sys.stderr)
        sys.exit(1)

    print(f"Calling: POST {BASE_URL}/route")
    print(f"  phone_number={phone!r}, lead_source={lead_source!r}\n")

    if BASE_URL.startswith("https://") and not os.environ.get("SNOWFLAKE_TOKEN"):
        key_path = os.environ.get("PRIVATE_KEY_PATH", "rsa_key.p8")
        if not os.path.isfile(key_path):
            print(f"Snowflake endpoint requires auth. Put your RSA private key at {key_path!r} (or set PRIVATE_KEY_PATH).")
            sys.exit(1)
        if not get_snowflake_token():
            print("Could not get OAuth token. Check PRIVATE_KEY_PATH, PyJWT/cryptography, SF_* env.")
            sys.exit(1)

    start = time.time()
    result = call_route(phone, lead_source)
    elapsed = time.time() - start

    if show_full_json:
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("success") else 1)

    print("--- Result ---")
    print(f"  success: {result.get('success')}")
    print(f"  action:  {result.get('action')}")
    print(f"  phone:   {result.get('phone_number', phone)}")
    if result.get("action") == "reroute":
        print(f"  reason:  {result.get('reason', 'age_band')}")
    if result.get("action") == "model":
        print(f"  model:   {result.get('model')}")
        pred = result.get("prediction")
        if pred and isinstance(pred.get("data"), list) and pred["data"]:
            row = pred["data"][0]
            if isinstance(row, (list, tuple)) and len(row) >= 2 and isinstance(row[1], dict):
                feat = row[1]
                print(f"  PREDICTION:       {feat.get('PREDICTION')}")
                print(f"  PREDICTION_PROBA: {feat.get('PREDICTION_PROBA')}")
                print(f"  tier:             {feat.get('tier')}")
    r = result.get("routing") or {}
    print(f"  age: {r.get('age')}  age_band: {r.get('age_band')}  gender: {r.get('gender')}  is_roku: {r.get('is_roku')}")
    if r.get("tu_duration_ms") is not None:
        print(f"  timing: TransUnion={r.get('tu_duration_ms')}ms", end="")
        if r.get("inference_duration_ms") is not None:
            print(f"  inference={r.get('inference_duration_ms')}ms")
        else:
            print()
    print(f"  (elapsed: {elapsed:.2f}s)")
    print("---")

    if not result.get("success"):
        err = result.get("error", result)
        print("Error:", err)
        if result.get("_auth_hint"):
            print("\n", result["_auth_hint"])
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

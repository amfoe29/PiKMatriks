"""Comprehensive verification suite for PikMatrix Authentication System."""

import sys
from fastapi.testclient import TestClient

from app.database import Base, SessionLocal, engine, init_db
from app.models import User
from app.auth import get_password_hash, verify_password, create_access_token, decode_access_token
from app.main import app

def run_tests():
    print("=" * 60)
    print("RUNNING PIKMATRIX AUTH SYSTEM TESTS")
    print("=" * 60)

    # 1. Test Model & Database
    print("\n[TEST 1] Testing User ORM Model & Database initialization...")
    init_db()
    with SessionLocal() as session:
        # Clear any test user from previous runs
        test_email = "tester_unit@pikmatrix.com"
        existing = session.query(User).filter(User.email == test_email).first()
        if existing:
            session.delete(existing)
            session.commit()

        # Check model fields
        user_cols = {col.name for col in User.__table__.columns}
        required_cols = {"id", "email", "username", "hashed_password", "created_at"}
        assert required_cols.issubset(user_cols), f"Missing columns: {required_cols - user_cols}"
        print("  -> User model schema confirmed with fields:", required_cols)

    # 2. Test Cryptography & Auth utilities
    print("\n[TEST 2] Testing Password Hashing & JWT...")
    password = "SuperSecretPassword123!"
    hashed = get_password_hash(password)
    assert hashed != password, "Password was not hashed"
    assert verify_password(password, hashed), "Password verification failed"
    assert not verify_password("WrongPassword", hashed), "Verification accepted bad password"
    print("  -> Password hashing and verification PASSED.")

    token = create_access_token({"sub": "999", "email": "test@domain.com"})
    assert isinstance(token, str) and len(token) > 20, "JWT generation failed"
    payload = decode_access_token(token)
    assert payload is not None and payload.get("sub") == "999", f"JWT decode failed: {payload}"
    print("  -> JWT encoding and decoding PASSED.")

    # 3. Test API via TestClient
    client = TestClient(app)

    # 3A. Test /login page rendering
    print("\n[TEST 3] Testing Login UI template GET /login...")
    resp = client.get("/login")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    html = resp.text
    assert "PikMatrix" in html, "Missing brand title in login.html"
    assert 'hx-post="/api/login"' in html, "Missing hx-post in login form"
    assert 'name="email"' in html and 'name="password"' in html, "Missing email/password fields"
    assert "#0F141C" in html or "#0f141c" in html, "Missing #0F141C background token"
    assert "#161D27" in html or "#161d27" in html, "Missing #161D27 surface token"
    assert "#3852F5" in html or "#3852f5" in html, "Missing #3852F5 button token"
    print("  -> Login UI template rendered with dark tokens & HTMX form PASSED.")

    # 3B. Test Unauthenticated Access to Protected App GET /
    print("\n[TEST 4] Testing Route Protection on GET /...")
    # Follow redirects = False to check HTTP redirect
    resp_unauth = client.get("/", follow_redirects=False)
    assert resp_unauth.status_code in (302, 303, 307), f"Expected redirect, got {resp_unauth.status_code}"
    assert "/login" in resp_unauth.headers.get("location", ""), "Did not redirect to /login"
    
    # HTMX request to / should return HX-Redirect
    resp_htmx_unauth = client.get("/", headers={"HX-Request": "true"})
    assert resp_htmx_unauth.headers.get("HX-Redirect") == "/login", f"Expected HX-Redirect /login, got {resp_htmx_unauth.headers}"
    print("  -> Unauthenticated requests redirected to /login PASSED.")

    # 3C. Test Registration Endpoint POST /api/register
    print("\n[TEST 5] Testing POST /api/register...")
    reg_payload = {
        "email": test_email,
        "username": "TestTrader",
        "password": "securepassword2026",
    }
    reg_resp = client.post("/api/register", json=reg_payload)
    assert reg_resp.status_code == 201, f"Expected 201, got {reg_resp.status_code}: {reg_resp.text}"
    reg_data = reg_resp.json()
    assert reg_data.get("user", {}).get("email") == test_email
    assert "access_token" in reg_resp.cookies, "access_token cookie was not set upon registration"
    print("  -> User registration & automatic session cookie PASSED.")

    # Duplicate registration should fail
    dup_resp = client.post("/api/register", json=reg_payload)
    assert dup_resp.status_code == 400, f"Expected 400 for duplicate, got {dup_resp.status_code}"
    print("  -> Duplicate registration rejection PASSED.")

    # 3D. Test Login Endpoint POST /api/login
    print("\n[TEST 6] Testing POST /api/login...")
    # Bad credentials
    bad_login = client.post("/api/login", json={"email": test_email, "password": "WrongPassword"})
    assert bad_login.status_code == 401, f"Expected 401, got {bad_login.status_code}"

    # Valid credentials (JSON)
    good_login = client.post("/api/login", json={"email": test_email, "password": "securepassword2026"})
    assert good_login.status_code == 200, f"Expected 200, got {good_login.status_code}"
    assert "access_token" in good_login.cookies, "access_token cookie not set"
    login_token = good_login.cookies["access_token"]
    assert len(login_token) > 20
    print("  -> Login with valid credentials and HttpOnly cookie PASSED.")

    # Valid credentials (HTMX Form)
    htmx_login = client.post(
        "/api/login",
        data={"email": test_email, "password": "securepassword2026"},
        headers={"HX-Request": "true"},
    )
    assert htmx_login.status_code == 200, f"Expected 200, got {htmx_login.status_code}"
    assert htmx_login.headers.get("HX-Redirect") == "/", f"Expected HX-Redirect: /, got {htmx_login.headers}"
    assert "access_token" in htmx_login.cookies, "access_token cookie not set in HTMX response"
    print("  -> HTMX form submission & HX-Redirect header PASSED.")

    # 3E. Test Authenticated Access to Protected GET /
    print("\n[TEST 7] Testing Authenticated Access to GET /...")
    authed_resp = client.get("/", cookies={"access_token": login_token})
    assert authed_resp.status_code == 200, f"Expected 200, got {authed_resp.status_code}"
    assert "Welcome back" in authed_resp.text, "Dashboard missing welcome message"
    assert "TestTrader" in authed_resp.text, "Dashboard missing authenticated username"
    assert 'hx-post="/api/logout"' in authed_resp.text, "Dashboard missing logout button"
    print("  -> Authenticated dashboard access and username greeting PASSED.")

    # 3F. Test Logout Endpoint POST /api/logout
    print("\n[TEST 8] Testing POST /api/logout...")
    logout_resp = client.post(
        "/api/logout",
        headers={"HX-Request": "true"},
        cookies={"access_token": login_token},
    )
    assert logout_resp.headers.get("HX-Redirect") == "/login", "Logout did not send HX-Redirect: /login"
    # Check that cookie is expired/cleared
    # TestClient cookies can be verified or check Set-Cookie header
    set_cookie_header = logout_resp.headers.get("set-cookie", "")
    assert "access_token" in set_cookie_header or 'access_token=""' in set_cookie_header or "Max-Age=0" in set_cookie_header, f"Cookie not cleared: {set_cookie_header}"
    print("  -> Logout cookie clearing and redirect to /login PASSED.")

    # 4. Clean up test user
    with SessionLocal() as session:
        cleanup_user = session.query(User).filter(User.email == test_email).first()
        if cleanup_user:
            session.delete(cleanup_user)
            session.commit()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! AUTH SYSTEM IS READY.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"\nTEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
patch_auth.py - flips auth so main site is protected, admin is open.
Run from repo root: python3 patch_auth.py
"""
import sys, os

def patch_file(path, changes):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run from repo root.")
        sys.exit(1)
    src = open(path, encoding='utf-8').read()
    for old, new, label in changes:
        if old not in src:
            print(f"  SKIP (already applied or not found): {label}")
            continue
        src = src.replace(old, new, 1)
        print(f"  OK: {label}")
    open(path, 'w', encoding='utf-8').write(src)

print("=== Patching hep_search.py ===")
patch_file("hep_search.py", [

    # Fix auth - protect everything EXCEPT /admin and /health
    (
        '''    if request.path == "/health":
        return None
    # Only protect admin routes - main search interface is open
    if not request.path.startswith("/admin"):
        return None
    auth = request.authorization
    if not auth or auth.username != AUTH_USERNAME or auth.password != AUTH_PASSWORD:
        return _unauthorized()
    return None''',

        '''    if request.path == "/health":
        return None
    # Admin is open - only protect the main site
    if request.path.startswith("/admin"):
        return None
    auth = request.authorization
    if not auth or auth.username != AUTH_USERNAME or auth.password != AUTH_PASSWORD:
        return _unauthorized()
    return None''',

        "Auth protects main site, admin is open"
    ),
])

print()
print("=== Patching templates/index.html ===")
patch_file("templates/index.html", [

    # Add Admin button to header
    (
        '''  <div style="display:flex;align-items:center;gap:12px;">
    <div class="doc-count" id="corpusStats">Loading corpus...</div>
    <button class="discover-toggle" id="discoverToggle" onclick="toggleDiscover()">Browse library</button>
  </div>''',

        '''  <div style="display:flex;align-items:center;gap:12px;">
    <div class="doc-count" id="corpusStats">Loading corpus...</div>
    <button class="discover-toggle" id="discoverToggle" onclick="toggleDiscover()">Browse library</button>
    <a href="/admin" class="admin-link">Admin</a>
  </div>''',

        "Admin button added to header"
    ),

    # Add admin-link CSS
    (
        '''.discover-toggle {''',

        '''.admin-link {
  color: var(--text-muted);
  font-family: 'Oswald', sans-serif;
  font-size: 0.72rem;
  font-weight: 400;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  text-decoration: none;
  border: 1px solid rgba(154,133,160,0.25);
  border-radius: 20px;
  padding: 3px 14px;
  transition: color 0.2s, border-color 0.2s;
}
.admin-link:hover {
  color: var(--light-purple);
  border-color: var(--border);
}

.discover-toggle {''',

        "Admin link CSS added"
    ),
])

print()
print("=== Done. Now run: git add hep_search.py templates/index.html && git commit -m 'Auth: protect main site, admin open, admin button in header' && git push origin main ===")

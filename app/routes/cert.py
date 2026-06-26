"""Routes for serving the shared MOBIUS.PROXY CA certificate.

Per intercom MSG-346 (2026-06-23): every mobius.* app exposes /install-cert
so a user can install the shared "NVR Local CA" from any project's UI.
After install, every mobius.* hostname is trusted (one cert, SAN includes
all hostnames, signed by this CA — no per-app cert juggling).

Cert is bind-mounted into the container at /etc/ssl/mobius_ca.pem
(docker-compose.yml). If the mount is missing, the route returns 503 with
a clear message rather than 404.
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["cert"])

# Where the bind-mount lands inside the container.
_CA_PATH = "/etc/ssl/mobius_ca.pem"
_CA_FILENAME = "mobius-local-ca.pem"


def _cert_available() -> bool:
    return os.path.isfile(_CA_PATH) and os.path.getsize(_CA_PATH) > 0


@router.get("/install-cert", response_class=HTMLResponse, include_in_schema=False)
async def install_cert_page():
    """Small landing page with download button + brief platform notes.
    Mirrors the per-app pattern (matches schola/nvr UX expectation per MSG-346)."""
    available = _cert_available()
    download_link = (
        f'<a href="/install-cert/download" class="btn-primary" download="{_CA_FILENAME}" '
        f'style="display:inline-block;padding:10px 18px;background:#0969da;color:#fff;'
        f'text-decoration:none;border-radius:6px;font-weight:600">⬇ Download certificate</a>'
        if available else
        '<p style="color:#e85a4f">Certificate is not mounted into this container. '
        'Operator: bind-mount <code>~/0_MOBIUS.PROXY/certs/ca.pem</code> at '
        '<code>/etc/ssl/mobius_ca.pem</code> and restart anamnesis-app.</p>'
    )
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Install MOBIUS Local Certificate · Anamnesis</title>
<style>
  body {{ background:#0d1117; color:#cfd5dd; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:40px 20px; line-height:1.6; }}
  .wrap {{ max-width:680px; margin:0 auto; }}
  h1 {{ margin-top:0; }} h2 {{ color:#8b95a0; font-size:14px; text-transform:uppercase; letter-spacing:0.06em; margin-top:32px; }}
  code, pre {{ background:#161b22; padding:2px 6px; border-radius:3px; font-family:'JetBrains Mono',Consolas,monospace; font-size:12.5px; }}
  pre {{ padding:10px 14px; overflow-x:auto; }}
  a {{ color:#58a6ff; }}
  .ok {{ color:#3fb950; }} .warn {{ color:#d29922; }}
  .panel {{ background:#0f1418; border:1px solid #30363d; border-radius:8px; padding:20px; margin:18px 0; }}
</style>
</head><body><div class="wrap">
<h1>🛡 Install the MOBIUS Local Certificate</h1>
<p>This is the shared CA cert that signs the TLS certificate for every
<code>https://mobius.*</code> hostname (anamnesis, schola, smarthome, tiles, nvr, &hellip;).
Install it once per device; every mobius.* page becomes trusted with no per-app re-import.</p>

<div class="panel">{download_link}</div>

<h2>Why install it</h2>
<p>Without this CA in your device's trust store, browsers show a security warning
on <code>https://mobius.anamnesis</code> (and the other mobius.* hostnames).
After install, all of them are trusted as a single bundle.</p>

<h2>How to install</h2>
<p><strong>macOS:</strong> double-click the downloaded <code>.pem</code> → Keychain Access opens → drag into <em>System</em> keychain → set "Always Trust".</p>
<p><strong>Windows:</strong> double-click → "Install Certificate" → Local Machine → "Place all certificates in the following store" → <em>Trusted Root Certification Authorities</em>.</p>
<p><strong>Linux (Debian/Ubuntu):</strong> <code>sudo cp mobius-local-ca.pem /usr/local/share/ca-certificates/mobius-local-ca.crt &amp;&amp; sudo update-ca-certificates</code>.</p>
<p><strong>iOS:</strong> AirDrop / email the file to yourself → tap → Settings prompts to install profile → Settings → General → About → Certificate Trust Settings → toggle ON for "NVR Local CA".</p>
<p><strong>Android:</strong> Settings → Security → Encryption &amp; credentials → Install a certificate → CA certificate.</p>

<h2>After install</h2>
<p>Visit <a href="https://mobius.anamnesis/">https://mobius.anamnesis/</a> — no warning. Same for the other mobius.* hostnames.</p>

<p style="font-size:12px;color:#8b95a0;margin-top:40px">
The CA is named "NVR Local CA" (legacy naming, kept stable to avoid forcing re-imports on every device).
Cert source: <code>~/0_MOBIUS.PROXY/certs/ca.pem</code> on dellserver.
Per intercom MSG-346 / MSG-347.
</p>
</div></body></html>""")


@router.get("/install-cert/download", include_in_schema=False)
async def install_cert_download():
    """Direct cert download. Served as application/x-x509-ca-cert so browsers
    treat it as a cert install on most platforms; the explicit `download`
    attribute on the link gives the user the .pem file regardless."""
    if not _cert_available():
        raise HTTPException(
            status_code=503,
            detail=f"Certificate not mounted at {_CA_PATH}. "
                   f"Bind-mount ~/0_MOBIUS.PROXY/certs/ca.pem into the container."
        )
    return FileResponse(
        _CA_PATH,
        media_type="application/x-x509-ca-cert",
        filename=_CA_FILENAME,
    )

# Feature Specification: Phase 5 — Embeddable Widget

**Feature Branch**: `phase-5-widget`

**Created**: 2026-05-18

**Status**: Draft

## User Scenarios & Testing

### User Story 1 — Host App Embeds The Widget With A Single Script Tag (Priority: P1)

A developer adds one `<script>` tag with a `data-widget-id` to their HTML page. The loader script injects an iframe pointing at the React widget bundle. The widget loads, reads its config, and displays a chat bubble.

**Why this priority**: The embed flow is the production-shaped surface and is demoed on Friday. Without it, the widget is not embeddable.

**Independent Test**: Open `demo/host/index.html` in a browser, confirm the chat bubble appears, expand it, and send a message. Verify the widget styled itself using the config fetched at load time.

**Acceptance Scenarios**:

1. **Given** a host page with the `<script>` tag and a valid `data-widget-id`, **When** the page loads, **Then** the loader injects an iframe and the chat bubble appears.
2. **Given** the widget is loaded, **When** the config is fetched, **Then** the widget applies the theme (primary color, position) from the database — not hardcoded values.
3. **Given** the widget is expanded, **When** the user sends a message, **Then** it is streamed back through the same FastAPI backend used by the Streamlit app.

---

### User Story 2 — Origin Allowlisting Blocks Unauthorized Hosts (Priority: P1)

The widget only loads on origins listed in its `allowed_origins` database field. An unauthorized host cannot iframe the widget — the browser blocks it via `Content-Security-Policy`.

**Why this priority**: Origin allowlisting is the production security practice required by the spec and demoed on Friday.

**Independent Test**: Load the widget on `allowed-host.local` — it works. Load it on `evil-host.local` — the browser blocks the embed. Both must be shown using real browser console output on Friday.

**Acceptance Scenarios**:

1. **Given** a host origin that is in `allowed_origins`, **When** the widget is embedded, **Then** it loads and functions correctly.
2. **Given** a host origin NOT in `allowed_origins`, **When** the widget is embedded, **Then** the browser blocks the iframe due to `Content-Security-Policy: frame-ancestors` and the widget does not load.
3. **Given** a CORS request from an allowed origin, **When** the API processes it, **Then** the `Access-Control-Allow-Origin` header matches the request origin.
4. **Given** a CORS request from a disallowed origin, **When** the API processes it, **Then** the CORS headers are absent and the browser blocks the request.

---

### User Story 3 — Admin Configures Widget And Gets Embed Snippet (Priority: P2)

An admin opens the Streamlit config page, creates a new widget (sets theme, greeting, allowed origins, enabled tools), and copies the generated `<script>` embed snippet.

**Why this priority**: Widget configuration drives the runtime behaviour of every embedded instance.

**Independent Test**: Create a widget via the admin page, copy the snippet, paste it into the demo host, and verify the widget reflects the configured theme and greeting.

**Acceptance Scenarios**:

1. **Given** an admin, **When** they create a widget config, **Then** a unique `widget_id` is generated and stored in the `widget` table.
2. **Given** a `widget_id`, **When** the admin views the config page, **Then** the embed snippet (`<script src="/widget.js" data-widget-id="...">`) is displayed and copyable.
3. **Given** a widget with `enabled_tools: ["classify", "rag_search"]`, **When** the widget is loaded, **Then** only those tools are available to the LLM for that widget session.

---

### Edge Cases

- What if the `data-widget-id` does not exist in the database? The loader must display a friendly error inside the iframe — not a blank white box.
- What if the React bundle fails to load (network error)? The loader must catch and log the error without crashing the host page.
- What if `postMessage` events arrive from an unexpected origin? The widget must validate the origin before processing.
- What if the widget bundle size exceeds the lean target? Audit bundle with a tool and document the trade-offs in `DECISIONS.md`.

## Requirements

### Functional Requirements

- **FR-001**: The React widget MUST be a standalone app built with Vite, output to a single bundled JS file.
- **FR-002**: The bundle MUST be served from the API or MinIO with proper cache headers. Bundle size (gzipped) MUST be reported in the submission block.
- **FR-003**: The widget MUST include: collapsed chat bubble, expandable chat panel, input box, streamed message display.
- **FR-004**: A `postMessage` channel MUST exist between widget and host — at minimum for iframe resize.
- **FR-005**: Theme (primary color, position) MUST come from the widget config fetched at runtime — NOT hardcoded.
- **FR-006**: A loader script MUST be served from `/widget.js`. It reads `data-widget-id` and injects the iframe.
- **FR-007**: The widget MUST fetch its config at load time and apply theme and greeting before displaying.
- **FR-008**: CORS `allowed_origins` MUST be enforced from the `widget` database table — NOT from a hardcoded env var.
- **FR-009**: The embed route MUST set `Content-Security-Policy: frame-ancestors <allowed_origins>`.
- **FR-010**: A `demo/host/` folder MUST contain a static host page (served by an nginx container) that embeds the widget.
- **FR-011**: The Friday demo MUST show: widget loading on an allowed host AND being blocked on a disallowed host, using real browser console/network output.
- **FR-012**: Widget config changes MUST be recorded in the audit log.

### Key Entities

- **Widget**: widget_id (public UUID), allowed_origins (string[]), theme (primary_color, position), greeting (str), enabled_tools (string[]), created_by (admin user id).
- **WidgetSession**: widget_id, user_token (anon or JWT), conversation_id.

## Success Criteria

- **SC-001**: A single `<script>` tag embeds the widget on any allowed host with no other setup.
- **SC-002**: Widget correctly applies runtime theme from database config.
- **SC-003**: Disallowed host is blocked by the browser — demonstrated with real console output.
- **SC-004**: Bundle size (gzipped) is reported in the submission block and kept lean.
- **SC-005**: `postMessage` iframe resize works in the demo host.
- **SC-006**: Admin can create, edit, and view widget configs and embed snippets in the Streamlit app.

## Assumptions

- The React bundle does not include the LLM or any ML model — it is purely a chat UI that calls the FastAPI backend.
- Anonymous widget sessions are supported (no login required to use the embedded widget).
- The demo host runs as an nginx container defined in `docker-compose.yml`.
- Tailwind CSS or vanilla CSS — choice is made before implementation and kept consistent.

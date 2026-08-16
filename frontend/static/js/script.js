/* ==========================================================================
   AI-Powered Smart Email Classifier — Frontend logic (SPA)
   ========================================================================== */

const API = {
    predict: "/api/predict",
    history: "/api/history",
    training: "/api/training-summary",
    health: "/api/health",
};

/* ---------- Toast ---------- */
function showToast(message) {
    let toast = document.getElementById("toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "toast";
        toast.className = "toast";
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.remove("show"), 2600);
}

/* ---------- Helpers ---------- */
function urgencyClass(value) {
    const v = String(value || "").toLowerCase();
    if (v === "high") return "high";
    if (v === "medium") return "medium";
    return "low";
}

function escapeHtml(str) {
    return String(str == null ? "" : str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

/* ---------- Result rendering ---------- */
function renderResult(data) {
    const panel = document.getElementById("result-panel");
    if (!panel) return;

    const urg = urgencyClass(data.urgency);

    panel.innerHTML = `
        <div class="result-head">
            <span class="badge category">${escapeHtml(data.category)}</span>
            <span class="badge ${urg}">${escapeHtml(data.urgency)} urgency</span>
        </div>

        <div class="conf-row">
            <div class="conf-label">
                <span><strong>Category confidence</strong></span>
                <span class="mono">${Number(data.category_confidence).toFixed(2)}%</span>
            </div>
            <div class="bar">
                <div class="fill" data-fill="${Number(data.category_confidence)}"></div>
            </div>
        </div>

        <div class="conf-row">
            <div class="conf-label">
                <span><strong>Urgency confidence</strong></span>
                <span class="mono">${Number(data.urgency_confidence).toFixed(2)}%</span>
            </div>
            <div class="bar">
                <div class="fill urgency-fill" data-fill="${Number(data.urgency_confidence)}"></div>
            </div>
        </div>

        <p class="muted" style="margin:14px 0 0;font-size:0.82rem;">
            Classified at <span class="mono">${escapeHtml(data.timestamp || "")}</span>
        </p>
    `;

    requestAnimationFrame(() => {
        panel.querySelectorAll(".fill[data-fill]").forEach((bar) => {
            const p = Number.parseFloat(bar.dataset.fill || "0");
            const safe = Number.isFinite(p) ? Math.max(0, Math.min(100, p)) : 0;
            bar.style.width = `${safe}%`;
        });
    });
}

/* ---------- Dashboard rendering ---------- */
function renderSummary(summary) {
    const set = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };
    set("m-total", summary.total ?? 0);
    set("m-high", `${summary.high_count ?? 0} (${summary.high_rate ?? 0}%)`);
    set("m-avg-cat", `${summary.avg_cat_conf ?? 0}%`);
    set("m-avg-urg", `${summary.avg_urg_conf ?? 0}%`);
}

function renderHistory(history) {
    const tbody = document.getElementById("history-body");
    if (!tbody) return;

    if (!history || history.length === 0) {
        tbody.innerHTML =
            '<tr><td colspan="6" class="muted" style="text-align:center;">No predictions yet.</td></tr>';
        return;
    }

    tbody.innerHTML = history
        .map((item) => {
            const urg = urgencyClass(item.urgency);
            return `
            <tr>
                <td class="mono">${escapeHtml(item.timestamp)}</td>
                <td>${escapeHtml(item.email)}</td>
                <td><span class="badge category">${escapeHtml(item.category)}</span></td>
                <td class="mono">${Number(item.category_confidence).toFixed(2)}%</td>
                <td><span class="badge ${urg}">${escapeHtml(item.urgency)}</span></td>
                <td class="mono">${Number(item.urgency_confidence).toFixed(2)}%</td>
            </tr>`;
        })
        .join("");
}

function renderTraining(summary) {
    const set = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };
    set("t-rows", summary.training_rows ?? 0);
    set("t-acc", `${summary.accuracy ?? 0}%`);
    set("t-max-iter", summary.model_max_iter ?? "—");
    set("t-class-weight", summary.model_class_weight ?? "—");
    set("t-max-features", summary.vectorizer_max_features ?? "—");

    const dist = document.getElementById("dist-body");
    if (dist) {
        const entries = Object.entries(summary.distribution || {});
        dist.innerHTML = entries.length
            ? entries
                  .map(
                      ([label, count]) => `
                <tr>
                    <td><span class="badge ${urgencyClass(label)}">${escapeHtml(label)}</span></td>
                    <td>${count}</td>
                </tr>`
                  )
                  .join("")
            : '<tr><td colspan="2" class="muted">No data.</td></tr>';
    }

    const metrics = document.getElementById("class-metrics-body");
    if (metrics) {
        const rows = summary.class_metrics || [];
        metrics.innerHTML = rows.length
            ? rows
                  .map(
                      (r) => `
                <tr>
                    <td><span class="badge ${urgencyClass(r.label)}">${escapeHtml(r.label)}</span></td>
                    <td>${r.precision}%</td>
                    <td>${r.recall}%</td>
                    <td>${r.f1}%</td>
                    <td>${r.support}</td>
                </tr>`
                  )
                  .join("")
            : '<tr><td colspan="5" class="muted">No data.</td></tr>';
    }
}

/* ---------- Data loading ---------- */
async function loadDashboard() {
    try {
        const [histRes, trainRes] = await Promise.all([
            fetch(API.history),
            fetch(API.training),
        ]);
        if (histRes.ok) {
            const histData = await histRes.json();
            renderSummary(histData.summary);
            renderHistory(histData.history);
        }
        if (trainRes.ok) {
            renderTraining(await trainRes.json());
        }
    } catch (err) {
        console.error("Failed to load dashboard:", err);
    }
}

/* ---------- Classify ---------- */
async function classifyEmail() {
    const input = document.getElementById("email-input");
    const btn = document.getElementById("classify-btn");
    const text = (input.value || "").trim();

    if (!text) {
        showToast("Please enter some email text first.");
        input.focus();
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Classifying…';

    try {
        const res = await fetch(API.predict, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: text }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || "Prediction failed.");
        }

        const data = await res.json();
        renderResult(data);
        await loadDashboard();
        const panel = document.getElementById("result-panel");
        if (panel) panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
        showToast(err.message || "Something went wrong.");
    } finally {
        btn.disabled = false;
        btn.innerHTML = "Classify Email";
    }
}

/* ---------- Init ---------- */
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("classify-form");
    if (form) {
        form.addEventListener("submit", (e) => {
            e.preventDefault();
            classifyEmail();
        });
    }

    const clearBtn = document.getElementById("clear-btn");
    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            const input = document.getElementById("email-input");
            if (input) input.value = "";
            const panel = document.getElementById("result-panel");
            if (panel) {
                panel.innerHTML =
                    '<div class="result-empty"><span class="emoji">📬</span>Your classification result will appear here.</div>';
            }
        });
    }

    document.querySelectorAll(".chip[data-sample]").forEach((chip) => {
        chip.addEventListener("click", () => {
            const input = document.getElementById("email-input");
            if (input) input.value = chip.dataset.sample;
            input.focus();
        });
    });

    loadDashboard();
});

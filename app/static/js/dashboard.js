/* ─── Anamnesis Dashboard JS (jQuery) ────────────────────────── */

$(function () {

    // ─── Status Banner ──────────────────────────────────────────
    function refresh_banner() {
        var el = $("#status-banner");
        el.addClass("refreshing");
        $.getJSON("/api/status/summary", function (data) {
            el.removeClass("refreshing").text(data.text || "—");
        }).fail(function () {
            el.removeClass("refreshing");
        });
    }
    refresh_banner();
    setInterval(refresh_banner, 60000);

    // ─── Tab Switching ──────────────────────────────────────────
    $(".tab").on("click", function () {
        var target_tab = $(this).data("tab");

        $(".tab").removeClass("active");
        $(this).addClass("active");

        $(".tab-content").removeClass("active");
        $("#tab-" + target_tab).addClass("active");

    });

    // ─── Episode List (Filter Tab) ──────────────────────────────
    $("#btn-filter-episodes").on("click", function () {
        var project_filter  = $("#filter-project").val().trim();
        var instance_filter = $("#filter-instance").val().trim();
        var tag_filter      = $("#filter-tag").val().trim();

        var query_params = [];
        if (project_filter)  query_params.push("project=" + encodeURIComponent(project_filter));
        if (instance_filter) query_params.push("instance=" + encodeURIComponent(instance_filter));
        if (tag_filter)      query_params.push("tag=" + encodeURIComponent(tag_filter));

        var url = "/api/episodes" + (query_params.length ? "?" + query_params.join("&") : "");

        $("#episodes-list").html('<p class="empty-state">Loading...</p>');

        $.getJSON(url, function (episodes) {
            if (episodes.length === 0) {
                $("#episodes-list").html('<p class="empty-state">No episodes found.</p>');
                return;
            }
            var html = "";
            episodes.forEach(function (ep) {
                html += render_episode_card(ep, false);
            });
            $("#episodes-list").html(html);
        }).fail(function (xhr) {
            $("#episodes-list").html(
                '<p class="empty-state">Error: ' + (xhr.responseJSON?.detail || xhr.statusText) + "</p>"
            );
        });
    });

    // ─── Vector Search ──────────────────────────────────────────
    $("#btn-search").on("click", function () {
        var query_text     = $("#search-query").val().trim();
        var top_k          = parseInt($("#search-top-k").val()) || 5;
        var project_filter = $("#search-project").val().trim();

        if (!query_text) {
            alert("Enter a search query.");
            return;
        }

        var payload = {
            query_text: query_text,
            top_k: top_k,
        };
        if (project_filter) payload.project_filter = project_filter;

        $("#search-results").html('<p class="empty-state">Searching...</p>');

        $.ajax({
            url: "/api/episodes/search",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify(payload),
            success: function (results) {
                if (results.length === 0) {
                    $("#search-results").html('<p class="empty-state">No matching episodes found.</p>');
                    return;
                }
                var html = "";
                results.forEach(function (ep) {
                    html += render_episode_card(ep, true);
                });
                $("#search-results").html(html);
            },
            error: function (xhr) {
                $("#search-results").html(
                    '<p class="empty-state">Error: ' + (xhr.responseJSON?.detail || xhr.statusText) + "</p>"
                );
            },
        });
    });

    // ─── Crawler Controls ─────────────────────────────────────────
    function load_crawler_status() {
        $.getJSON("/api/crawler/status", function (status) {
            var html =
                '<div class="kpi-grid">' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + (status.running ? "RUNNING" : "IDLE") + '</span>' +
                        '<span class="kpi-label">Status</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.total_episodes_ingested + '</span>' +
                        '<span class="kpi-label">Total Ingested</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.episodes_ingested_last_run + '</span>' +
                        '<span class="kpi-label">Last Run Ingested</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.last_run_duration_seconds + 's</span>' +
                        '<span class="kpi-label">Last Run Duration</span>' +
                    '</div>' +
                '</div>' +
                '<p><strong>Last run:</strong> ' + (status.last_run || "never") + '</p>' +
                '<p><strong>Interval:</strong> ' + status.interval_seconds + 's</p>';

            if (status.errors && status.errors.length) {
                html += '<h4>Errors (last run)</h4><ul>';
                status.errors.forEach(function (err) {
                    html += '<li style="color: var(--danger);">' + escape_html(err) + '</li>';
                });
                html += '</ul>';
            }

            $("#crawler-status").html(html);
        }).fail(function () {
            $("#crawler-status").html('<p class="empty-state">Failed to load crawler status.</p>');
        });
    }

    $("#btn-refresh-crawler").on("click", load_crawler_status);

    $("#btn-crawl-now").on("click", function () {
        if (!confirm("Run crawler now?\n\nThis uses local embeddings only (free, no API costs).\nIt may take a few minutes to scan all sources.")) return;
        var $btn = $(this);
        $btn.prop("disabled", true).text("Crawling...");

        $.ajax({
            url: "/api/crawler/run",
            method: "POST",
            success: function (result) {
                $btn.prop("disabled", false).text("Run Crawl Now");
                load_crawler_status();
                alert("Crawl complete: " + result.episodes_ingested + " episodes ingested.");
            },
            error: function (xhr) {
                $btn.prop("disabled", false).text("Run Crawl Now");
                alert("Crawl failed: " + (xhr.responseJSON?.detail || xhr.statusText));
            },
        });
    });

    // Auto-load crawler status when switching to crawler tab
    $(".tab[data-tab='crawler']").on("click", function () {
        load_crawler_status();
        load_crawler_schedule();
    });

    function load_crawler_schedule() {
        $.getJSON("/api/jsonl/schedule", function (data) {
            $("#crawler-schedule-select").val(data.crawler_schedule || "every_30m");
        });
    }

    $("#btn-save-crawler-schedule").on("click", function () {
        var schedule = $("#crawler-schedule-select").val();
        $.ajax({
            url: "/api/jsonl/schedule",
            method: "PUT",
            contentType: "application/json",
            data: JSON.stringify({crawler_schedule: schedule}),
            success: function () {
                $("#crawler-schedule-status")
                    .text("Saved!")
                    .css("color", "var(--success)");
                setTimeout(function () { $("#crawler-schedule-status").text(""); }, 3000);
            },
            error: function (xhr) {
                $("#crawler-schedule-status")
                    .text("Error: " + (xhr.responseJSON?.error || xhr.statusText))
                    .css("color", "var(--danger)");
            },
        });
    });

    // ─── JSONL Ingester ──────────────────────────────────────────
    function load_jsonl_status() {
        $.getJSON("/api/jsonl/status", function (status) {
            var html =
                '<div class="kpi-grid">' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + (status.running ? "RUNNING" : "IDLE") + '</span>' +
                        '<span class="kpi-label">Status</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.total_episodes_ingested + '</span>' +
                        '<span class="kpi-label">Total Ingested</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.episodes_ingested_last_run + '</span>' +
                        '<span class="kpi-label">Last Run Ingested</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.files_processed + '</span>' +
                        '<span class="kpi-label">Files Processed</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.exchanges_evaluated + '</span>' +
                        '<span class="kpi-label">Exchanges Evaluated</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.exchanges_skipped + '</span>' +
                        '<span class="kpi-label">Exchanges Skipped</span>' +
                    '</div>' +
                    '<div class="kpi-card">' +
                        '<span class="kpi-value">' + status.last_run_duration_seconds + 's</span>' +
                        '<span class="kpi-label">Last Run Duration</span>' +
                    '</div>' +
                '</div>' +
                '<p><strong>Last run:</strong> ' + (status.last_run || "never") + '</p>';

            if (status.last_interrupted) {
                html += '<p style="color: var(--warning);"><strong>Previous run interrupted:</strong> ' +
                    escape_html(status.last_interrupted) +
                    ' (no data corruption — dedup covers partial runs)</p>';
            }

            if (status.errors && status.errors.length) {
                html += '<h4>Errors (last run)</h4><ul>';
                status.errors.forEach(function (err) {
                    html += '<li style="color: var(--danger);">' + escape_html(err) + '</li>';
                });
                html += '</ul>';
            }

            $("#jsonl-status").html(html);
        }).fail(function () {
            $("#jsonl-status").html('<p class="empty-state">Failed to load JSONL ingester status.</p>');
        });
    }

    $("#btn-refresh-jsonl").on("click", load_jsonl_status);

    var jsonl_poll_timer = null;

    function start_jsonl_polling() {
        if (jsonl_poll_timer) return;
        jsonl_poll_timer = setInterval(function () {
            $.getJSON("/api/jsonl/status", function (status) {
                // Update tiles in-place
                var $panel = $("#jsonl-status");
                $panel.find(".kpi-value").eq(0).text(status.running ? "RUNNING" : "IDLE");
                $panel.find(".kpi-value").eq(1).text(status.total_episodes_ingested);
                $panel.find(".kpi-value").eq(2).text(status.episodes_ingested_last_run);
                $panel.find(".kpi-value").eq(3).text(status.files_processed);
                $panel.find(".kpi-value").eq(4).text(status.exchanges_evaluated);
                $panel.find(".kpi-value").eq(5).text(status.exchanges_skipped);
                $panel.find(".kpi-value").eq(6).text(status.last_run_duration_seconds + "s");

                if (!status.running) {
                    stop_jsonl_polling();
                    $("#btn-jsonl-ingest").prop("disabled", false).text("Run JSONL Ingestion");
                }
            });
        }, 2000);
    }

    function stop_jsonl_polling() {
        if (jsonl_poll_timer) {
            clearInterval(jsonl_poll_timer);
            jsonl_poll_timer = null;
        }
    }

    $("#btn-jsonl-ingest").on("click", function () {
        var backend = $("#jsonl-backend-select").val() || "ollama";
        var msg;
        if (backend === "claude") {
            msg = "Run JSONL ingestion now?\n\n" +
                  "WARNING: Claude API is selected — this WILL incur costs.\n" +
                  "Each exchange uses ~1-2K input tokens + ~300 output tokens.\n\n" +
                  "To run for free, switch to Ollama in Settings above.";
        } else {
            msg = "Run JSONL ingestion now?\n\n" +
                  "Summarization uses local Ollama (free, no API costs).\n" +
                  "This may take several minutes depending on backlog size.";
        }
        if (!confirm(msg)) return;
        var $btn = $(this);
        $btn.prop("disabled", true).text("Ingesting...");

        // Start live polling immediately
        load_jsonl_status();
        start_jsonl_polling();

        $.ajax({
            url: "/api/jsonl/ingest",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({}),
            success: function (result) {
                if (result.status === "already_running") {
                    alert("JSONL ingestion is already running.");
                }
                // Polling handles the rest — button re-enabled when status goes IDLE
            },
            error: function (xhr) {
                stop_jsonl_polling();
                $btn.prop("disabled", false).text("Run JSONL Ingestion");
                alert("JSONL ingestion failed: " + (xhr.responseJSON?.detail || xhr.statusText));
            },
        });
    });

    $(".tab[data-tab='jsonl']").on("click", function () {
        load_jsonl_status();
        load_jsonl_settings();
        // If it's already running (e.g. navigated away and back), resume polling
        $.getJSON("/api/jsonl/status", function (status) {
            if (status.running) start_jsonl_polling();
        });
    });

    // ─── JSONL Settings ──────────────────────────────────────────

    function load_jsonl_settings() {
        $.getJSON("/api/jsonl/settings", function (settings) {
            $("#jsonl-backend-select").val(settings.summarization_backend || "ollama");
            $("#jsonl-max-exchanges").val(settings.max_exchanges_per_run || 0);
            $("#jsonl-cpu-pct").val(settings.cpu_core_pct || 70);
            $("#jsonl-schedule-select").val(settings.schedule || "nightly");
            update_cpu_label();
            toggle_ollama_model_row();
            update_schedule_hint();

            // Load Ollama models for the selector
            load_jsonl_ollama_models(settings.ollama_model);
        });
    }

    function update_schedule_hint() {
        var backend = $("#jsonl-backend-select").val();
        if (backend === "claude") {
            $("#jsonl-schedule-hint").text("Schedule disabled: Claude API is a paid backend.").css("color", "var(--warning)");
            $("#jsonl-schedule-select").prop("disabled", true);
        } else {
            $("#jsonl-schedule-hint").text("Free backends only. Auto-disabled for paid models.").css("color", "");
            $("#jsonl-schedule-select").prop("disabled", false);
        }
    }

    var _model_registry = {};

    function load_jsonl_ollama_models(selected_model) {
        // Fetch registry first for metadata, then populate dropdown from Ollama
        $.getJSON("/api/jsonl/models", function (reg) {
            (reg.models || []).forEach(function (m) {
                _model_registry[m.model_id] = m;
            });
        }).always(function () {
            $.getJSON("/api/chat/models", function (data) {
                var $sel = $("#jsonl-ollama-model");
                $sel.empty();
                if (data.models && data.models.length) {
                    data.models.forEach(function (m) {
                        var info = _model_registry[m];
                        var label = info ? info.display_name + " — " + m : m;
                        var opt = $("<option>").val(m).text(label);
                        if (m === selected_model) opt.prop("selected", true);
                        $sel.append(opt);
                    });
                } else {
                    $sel.append($("<option>").val("").text("No models found"));
                }
                update_model_hint();
            });
        });
    }

    function update_model_hint() {
        var model_id = $("#jsonl-ollama-model").val();
        var info = _model_registry[model_id];
        if (info) {
            $("#jsonl-model-hint").text(info.parameters + " params · " + info.notes);
        } else {
            $("#jsonl-model-hint").text("");
        }
    }

    function toggle_ollama_model_row() {
        if ($("#jsonl-backend-select").val() === "ollama") {
            $("#jsonl-ollama-model-row").show();
        } else {
            $("#jsonl-ollama-model-row").hide();
        }
    }

    function update_cpu_label() {
        var pct = parseInt($("#jsonl-cpu-pct").val());
        $("#jsonl-cpu-pct-label").text(pct + "%");
        // Estimate cores (we can't know total from client, but show percentage)
        $("#jsonl-cpu-cores-hint").text("(of total logical cores)");
    }

    $("#jsonl-backend-select").on("change", function () {
        toggle_ollama_model_row();
        update_schedule_hint();

        if ($(this).val() === "claude") {
            alert(
                "Claude API selected.\n\n" +
                "WARNING: This will incur costs per exchange summarized.\n" +
                "Scheduled runs will be auto-disabled.\n\n" +
                "Switch back to Ollama to re-enable scheduling."
            );
            $("#jsonl-schedule-select").val("disabled");
        }
    });
    $("#jsonl-ollama-model").on("change", update_model_hint);
    $("#jsonl-cpu-pct").on("input", update_cpu_label);

    $("#btn-save-jsonl-settings").on("click", function () {
        var settings_payload = {
            summarization_backend: $("#jsonl-backend-select").val(),
            ollama_model: $("#jsonl-ollama-model").val() || undefined,
            max_exchanges_per_run: parseInt($("#jsonl-max-exchanges").val()) || 0,
            cpu_core_pct: parseInt($("#jsonl-cpu-pct").val()) || 70,
        };

        var schedule_payload = {
            jsonl_schedule: $("#jsonl-schedule-select").val(),
        };

        var save_count = 0;
        var save_errors = [];

        function check_done() {
            save_count++;
            if (save_count < 2) return;
            if (save_errors.length) {
                $("#jsonl-settings-status")
                    .text("Error: " + save_errors.join("; "))
                    .css("color", "var(--danger)");
            } else {
                $("#jsonl-settings-status")
                    .text("Saved!")
                    .css("color", "var(--success)");
                setTimeout(function () { $("#jsonl-settings-status").text(""); }, 3000);
            }
        }

        // Save settings
        $.ajax({
            url: "/api/jsonl/settings",
            method: "PUT",
            contentType: "application/json",
            data: JSON.stringify(settings_payload),
            success: function (result) {
                if (result._schedule_warning) {
                    alert(result._schedule_warning);
                    $("#jsonl-schedule-select").val("disabled");
                    update_schedule_hint();
                }
                check_done();
            },
            error: function (xhr) {
                save_errors.push(xhr.responseJSON?.error || xhr.statusText);
                check_done();
            },
        });

        // Save schedule
        $.ajax({
            url: "/api/jsonl/schedule",
            method: "PUT",
            contentType: "application/json",
            data: JSON.stringify(schedule_payload),
            success: check_done,
            error: function (xhr) {
                save_errors.push(xhr.responseJSON?.error || xhr.statusText);
                check_done();
            },
        });
    });

    // ─── Render Helper ──────────────────────────────────────────
    function render_episode_card(ep, show_score) {
        var tags_html = "";
        if (ep.tags && ep.tags.length) {
            ep.tags.forEach(function (tag) {
                tags_html += '<span class="tag">' + escape_html(tag) + "</span>";
            });
        }

        var score_html = "";
        if (show_score && ep.similarity_score !== undefined) {
            score_html = '<span class="similarity-score">' +
                (ep.similarity_score * 100).toFixed(1) + "% match</span>";
        }

        return (
            '<div class="episode-card">' +
                '<div class="episode-header">' +
                    '<span class="episode-id">' + escape_html(ep.episode_id) + "</span>" +
                    '<span class="episode-meta">' +
                        escape_html(ep.instance) + " / " + escape_html(ep.project) +
                    "</span>" +
                "</div>" +
                '<p class="episode-summary">' + escape_html(ep.summary) + "</p>" +
                '<div class="episode-footer">' +
                    '<span class="episode-tags">' + tags_html + "</span>" +
                    score_html +
                    '<span class="episode-retrievals">Retrieved: ' +
                        (ep.retrieval_count || 0) + "x</span>" +
                "</div>" +
            "</div>"
        );
    }

    function escape_html(str) {
        if (!str) return "";
        return $("<div>").text(str).html();
    }

    // ─── Chat Tab ────────────────────────────────────────────────

    var CHAT_SESSION_KEY = "anamnesis_chat_session";
    var CLAUDE_TIMEOUT_MS = 30 * 60 * 1000;  // 30 minutes

    var chat_state = {
        session_id: localStorage.getItem(CHAT_SESSION_KEY) || null,
        backend: "ollama",
        model: null,
        claude_enabled: false,
        claude_cli_enabled: false,
        anamnesis_enabled: false,
        anamnesis_available: false,
        anamnesis_unavailable_msg: "",
        claude_enabled_at: null,
        claude_timer_interval: null,
        balance_poll_interval: null,
        streaming: false,
    };

    // Load Ollama models when Chat tab opens
    $(".tab[data-tab='chat']").on("click", function () {
        load_ollama_models();
        if (!chat_state.session_id) {
            new_chat_session();
        }
    });

    function new_chat_session() {
        chat_state.session_id = "s-" + Math.random().toString(36).slice(2, 12);
        localStorage.setItem(CHAT_SESSION_KEY, chat_state.session_id);
    }

    // ── Ollama model selector ─────────────────────────────────────
    function load_ollama_models() {
        $.getJSON("/api/chat/models", function (data) {
            var $sel = $("#chat-model-select");
            $sel.empty();
            if (data.models && data.models.length) {
                data.models.forEach(function (m) {
                    var opt = $("<option>").val(m).text(m);
                    if (m === data.default) opt.prop("selected", true);
                    $sel.append(opt);
                });
                chat_state.model = $sel.val();
                $sel.show();
            } else {
                $sel.hide();
                var err = data.error || "No models found";
                $sel.append($("<option>").val("").text(err));
            }
        }).fail(function () {
            $("#chat-model-select").hide();
        });
    }

    $("#chat-model-select").on("change", function () {
        chat_state.model = $(this).val();
    });

    // ── Claude CLI toggle ($0 subscription) ──────────────────────
    $("#btn-claude-cli-toggle").on("click", function () {
        if (chat_state.claude_cli_enabled) {
            disable_claude_cli();
        } else {
            enable_claude_cli();
        }
    });

    function enable_claude_cli() {
        // Disable other backends first
        if (chat_state.claude_enabled)     disable_claude();
        if (chat_state.anamnesis_enabled)  disable_anamnesis();

        chat_state.claude_cli_enabled = true;
        chat_state.backend = "claude_cli";

        $("#btn-claude-cli-toggle").text("CLI ON — click to disable")
            .removeClass("btn-claude-off").addClass("btn-claude-on");
        $("#chat-backend-badge").text("Claude CLI ($0)")
            .removeClass("ollama-badge claude-badge").addClass("claude-badge");
        $("#chat-model-select").hide();
        show_chat_system("Claude CLI backend enabled. Uses subscription — $0 per message.");
    }

    function disable_claude_cli() {
        chat_state.claude_cli_enabled = false;
        chat_state.backend = "ollama";

        $("#btn-claude-cli-toggle").text("Claude CLI ($0)")
            .removeClass("btn-claude-on").addClass("btn-claude-off");
        $("#chat-backend-badge").text("Ollama")
            .removeClass("claude-badge").addClass("ollama-badge");
        $("#chat-model-select").show();
        load_ollama_models();
    }

    // ── AnamnesisGPT — machine-gated toggle ────────────────────────
    // Check availability on page load
    $.getJSON("/api/anamnesis-gpt/status", function (data) {
        chat_state.anamnesis_available = data.available;
        chat_state.anamnesis_unavailable_msg = data.message || "";
        var btn = $("#btn-anamnesis-toggle");
        btn.removeClass("btn-anamnesis-checking");
        if (data.available) {
            btn.prop("disabled", false);
        } else {
            btn.addClass("btn-anamnesis-unavailable");
        }
    }).fail(function () {
        $("#btn-anamnesis-toggle").removeClass("btn-anamnesis-checking")
            .addClass("btn-anamnesis-unavailable");
    });

    $("#btn-anamnesis-toggle").on("click", function () {
        if (!chat_state.anamnesis_available) {
            show_chat_system(chat_state.anamnesis_unavailable_msg ||
                "AnamnesisGPT is not available on this host. " +
                "Custom training on your own data is coming soon.");
            return;
        }
        if (chat_state.anamnesis_enabled) {
            disable_anamnesis();
        } else {
            enable_anamnesis();
        }
    });

    function enable_anamnesis() {
        if (chat_state.claude_enabled)     disable_claude();
        if (chat_state.claude_cli_enabled) disable_claude_cli();

        chat_state.anamnesis_enabled = true;
        chat_state.backend = "anamnesis";

        $("#btn-anamnesis-toggle").text("Anamnesis ON — click to disable")
            .removeClass("btn-claude-off").addClass("btn-claude-on");
        $("#chat-backend-badge").text("AnamnesisGPT")
            .removeClass("ollama-badge claude-badge").addClass("anamnesis-badge");
        $("#chat-model-select").hide();
        show_chat_system("AnamnesisGPT active — personal LLM trained on your data.");
    }

    function disable_anamnesis() {
        chat_state.anamnesis_enabled = false;
        chat_state.backend = "ollama";

        $("#btn-anamnesis-toggle").text("Anamnesis")
            .removeClass("btn-claude-on").addClass("btn-claude-off");
        $("#chat-backend-badge").text("Ollama")
            .removeClass("anamnesis-badge").addClass("ollama-badge");
        $("#chat-model-select").show();
        load_ollama_models();
    }

    // ── Claude toggle ─────────────────────────────────────────────
    $("#btn-claude-toggle").on("click", function () {
        if (chat_state.claude_enabled) {
            disable_claude();
        } else {
            enable_claude();
        }
    });

    function enable_claude() {
        // Disable other backends if active
        if (chat_state.claude_cli_enabled) disable_claude_cli();
        if (chat_state.anamnesis_enabled)  disable_anamnesis();

        chat_state.claude_enabled = true;
        chat_state.backend = "claude";
        chat_state.claude_enabled_at = Date.now();

        $("#btn-claude-toggle").text("Claude ON — Click to disable").removeClass("btn-claude-off").addClass("btn-claude-on");
        $("#chat-backend-badge").text("Claude API").removeClass("ollama-badge").addClass("claude-badge");
        $("#chat-model-select").hide();
        $("#chat-balance-group").show();
        $("#claude-timer-badge").show();

        fetch_balance();
        // Poll balance every 60s while Claude is active
        chat_state.balance_poll_interval = setInterval(fetch_balance, 60000);

        // Countdown timer display
        update_claude_timer();
        chat_state.claude_timer_interval = setInterval(function () {
            var elapsed = Date.now() - chat_state.claude_enabled_at;
            var remaining = CLAUDE_TIMEOUT_MS - elapsed;
            if (remaining <= 0) {
                auto_disable_claude();
            } else {
                update_claude_timer();
                // Warn at 5 minutes remaining
                if (remaining <= 5 * 60 * 1000 && remaining > (5 * 60 * 1000 - 10000)) {
                    show_chat_system("Claude API will auto-disable in 5 minutes.");
                }
            }
        }, 10000);
    }

    function disable_claude() {
        chat_state.claude_enabled = false;
        chat_state.backend = "ollama";

        clearInterval(chat_state.claude_timer_interval);
        clearInterval(chat_state.balance_poll_interval);
        chat_state.claude_timer_interval = null;
        chat_state.balance_poll_interval = null;

        $("#btn-claude-toggle").text("Use Claude API").removeClass("btn-claude-on").addClass("btn-claude-off");
        $("#chat-backend-badge").text("Ollama").removeClass("claude-badge").addClass("ollama-badge");
        $("#chat-model-select").show();
        $("#chat-balance-group").hide();
        $("#claude-timer-badge").hide();
        load_ollama_models();
    }

    function auto_disable_claude() {
        disable_claude();
        var confirmed = confirm(
            "Claude API has been automatically disabled after 30 minutes.\n\n" +
            "Click OK to re-enable, or Cancel to stay on Ollama."
        );
        if (confirmed) {
            enable_claude();
        } else {
            show_chat_system("Switched back to Ollama.");
        }
    }

    function update_claude_timer() {
        var elapsed = Date.now() - chat_state.claude_enabled_at;
        var remaining_s = Math.max(0, Math.floor((CLAUDE_TIMEOUT_MS - elapsed) / 1000));
        var m = Math.floor(remaining_s / 60);
        var s = remaining_s % 60;
        $("#claude-timer-badge").text("Auto-off in " + m + "m " + (s < 10 ? "0" : "") + s + "s");
    }

    // ── Balance ───────────────────────────────────────────────────
    function fetch_balance() {
        $.getJSON("/api/chat/balance", function (data) {
            if (data.available && data.data) {
                var d = data.data;
                // Try common field names Anthropic might return
                var amount = d.credits_remaining ?? d.balance ?? d.available_credits ?? null;
                if (amount !== null) {
                    $("#chat-balance-value").text("$" + parseFloat(amount).toFixed(2));
                } else {
                    // Show raw JSON abbreviated
                    $("#chat-balance-value").text(JSON.stringify(d).slice(0, 40));
                }
            } else {
                var reason = data.reason || "unavailable";
                if (data.console_url) {
                    $("#chat-balance-value").html(
                        '<a href="' + data.console_url + '" target="_blank" style="color:var(--accent)">Check console</a>'
                    );
                } else {
                    $("#chat-balance-value").text(reason);
                }
            }
        }).fail(function () {
            $("#chat-balance-value").text("error");
        });
    }

    $("#btn-refresh-balance").on("click", fetch_balance);

    // ── Message rendering ─────────────────────────────────────────
    function append_message(role, content, streaming_id) {
        var cls = role === "user" ? "chat-msg-user" : "chat-msg-assistant";
        var label = role === "user" ? "You" :
            (chat_state.backend === "anamnesis" ? "AnamnesisGPT" : "Anamnesis");
        var id_attr = streaming_id ? ' id="' + streaming_id + '"' : "";
        var html =
            '<div class="chat-msg ' + cls + '"' + id_attr + ">" +
                '<span class="chat-msg-label">' + escape_html(label) + "</span>" +
                '<div class="chat-msg-content">' + escape_html(content) + "</div>" +
            "</div>";
        $("#chat-messages").append(html);
        scroll_chat();
    }

    function show_chat_system(text) {
        var html = '<div class="chat-msg chat-msg-system"><em>' + escape_html(text) + "</em></div>";
        $("#chat-messages").append(html);
        scroll_chat();
    }

    function scroll_chat() {
        var $m = $("#chat-messages");
        $m.scrollTop($m[0].scrollHeight);
    }

    // ── Send ──────────────────────────────────────────────────────
    $("#btn-chat-send").on("click", do_send);

    $("#chat-input").on("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            do_send();
        }
    });

    function do_send() {
        if (chat_state.streaming) return;

        var msg = $("#chat-input").val().trim();
        if (!msg) return;

        if (!chat_state.session_id) new_chat_session();

        $("#chat-input").val("").prop("disabled", true);
        $("#btn-chat-send").prop("disabled", true);
        chat_state.streaming = true;

        append_message("user", msg);

        // Placeholder for streaming assistant response
        var stream_id = "stream-" + Date.now();
        var html =
            '<div class="chat-msg chat-msg-assistant" id="' + stream_id + '">' +
                '<span class="chat-msg-label">Anamnesis</span>' +
                '<div class="chat-tool-uses"></div>' +
                '<div class="chat-msg-content"><span class="chat-cursor">&#9646;</span></div>' +
            "</div>";
        $("#chat-messages").append(html);
        scroll_chat();

        var payload = JSON.stringify({
            message: msg,
            backend: chat_state.backend,
            model: chat_state.model || undefined,
            session_id: chat_state.session_id,
            top_k: 3,
            attached_files: attached_files.map(function (f) {
                return {path: f.path, source: f.source};
            }),
        });

        // Use fetch + ReadableStream for SSE
        fetch("/api/chat/stream", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: payload,
        }).then(function (resp) {
            if (!resp.ok) {
                throw new Error("HTTP " + resp.status);
            }
            var reader = resp.body.getReader();
            var decoder = new TextDecoder();
            var buffer = "";
            var accumulated = "";

            function read_chunk() {
                reader.read().then(function (result) {
                    if (result.done) {
                        finish_streaming();
                        return;
                    }
                    buffer += decoder.decode(result.value, {stream: true});
                    var lines = buffer.split("\n");
                    buffer = lines.pop();  // keep incomplete line

                    lines.forEach(function (line) {
                        if (!line.startsWith("data:")) return;
                        var raw = line.slice(5).trim();
                        if (!raw) return;
                        try {
                            var event = JSON.parse(raw);
                            if (event.token) {
                                accumulated += event.token;
                                $("#" + stream_id + " .chat-msg-content")
                                    .html(escape_html(accumulated) + '<span class="chat-cursor">▌</span>');
                                scroll_chat();
                            } else if (event.tool_use) {
                                var q = event.tool_use.query || "";
                                var indicator =
                                    '<div class="chat-tool-use-item">' +
                                        '<span class="chat-tool-use-icon">&#128269;</span>' +
                                        ' Searching memory: <em>' + escape_html(q) + '</em>' +
                                    '</div>';
                                $("#" + stream_id + " .chat-tool-uses").append(indicator);
                                scroll_chat();
                            } else if (event.error) {
                                $("#" + stream_id + " .chat-msg-content")
                                    .html('<span style="color:var(--danger)">' + escape_html(event.error) + "</span>");
                                finish_streaming();
                                return;
                            } else if (event.done) {
                                $("#" + stream_id + " .chat-msg-content").text(accumulated).removeClass("chat-streaming");
                                // Show cost badge
                                if (event.cost) {
                                    var cost_html =
                                        '<div class="chat-cost-badge">' +
                                        escape_html(event.cost) + "</div>";
                                    $("#" + stream_id).append(cost_html);
                                }
                                finish_streaming();
                                return;
                            }
                        } catch (e) { /* ignore parse errors */ }
                    });

                    read_chunk();
                }).catch(function (err) {
                    $("#" + stream_id + " .chat-msg-content")
                        .html('<span style="color:var(--danger)">Stream error: ' + escape_html(err.message) + "</span>");
                    finish_streaming();
                });
            }

            read_chunk();

        }).catch(function (err) {
            $("#" + stream_id + " .chat-msg-content")
                .html('<span style="color:var(--danger)">Error: ' + escape_html(err.message) + "</span>");
            finish_streaming();
        });
    }

    function finish_streaming() {
        chat_state.streaming = false;
        $("#chat-input").prop("disabled", false).focus();
        $("#btn-chat-send").prop("disabled", false);
        // Refresh sessions list so new/updated session appears
        load_chat_sessions();
    }

    // ── Clear conversation ────────────────────────────────────────
    $("#btn-clear-chat").on("click", function () {
        if (!confirm("Clear this conversation?")) return;
        if (chat_state.session_id) {
            $.ajax({
                url: "/api/chat/session/" + chat_state.session_id,
                method: "DELETE",
            });
        }
        $("#chat-messages").empty();
        new_chat_session();
        show_chat_system("Conversation cleared. New session started.");
    });

    // ── Session list ──────────────────────────────────────────────
    function load_chat_sessions() {
        $.getJSON("/api/chat/sessions", function (data) {
            var $list = $("#sessions-list");
            $list.empty();
            if (!data.sessions || !data.sessions.length) {
                $list.append('<p class="empty-state" style="font-size:0.8em;padding:8px;">No saved chats yet.</p>');
                return;
            }
            data.sessions.forEach(function (s) {
                $list.append(render_session_item(s));
            });
        });
    }

    function render_session_item(s) {
        var is_active = s.session_id === chat_state.session_id;
        var date = s.updated_at ? new Date(s.updated_at).toLocaleDateString() : "";
        var backend_label = s.backend || "ollama";
        var item = $(
            '<div class="session-item' + (is_active ? " active" : "") + '" data-sid="' + s.session_id + '">' +
            '<div class="session-item-title" title="Double-click to rename">' + escape_html(s.title || "Untitled") + '</div>' +
            '<div class="session-item-meta"><span>' + escape_html(backend_label) + '</span><span>' + date + '</span></div>' +
            '<button class="session-delete-btn" title="Delete">✕</button>' +
            '</div>'
        );

        // Click → load session
        item.on("click", function (e) {
            if ($(e.target).hasClass("session-delete-btn")) return;
            if ($(e.target).attr("contenteditable") === "true") return;
            load_chat_session(s.session_id);
        });

        // Double-click title → rename
        item.find(".session-item-title").on("dblclick", function (e) {
            e.stopPropagation();
            var $title = $(this);
            $title.attr("contenteditable", "true").focus();
            var range = document.createRange();
            range.selectNodeContents(this);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);

            $title.off("keydown.rename").on("keydown.rename", function (ev) {
                if (ev.key === "Enter") { ev.preventDefault(); $title.blur(); }
                if (ev.key === "Escape") { $title.text(s.title || "Untitled").blur(); }
            });
            $title.off("blur.rename").on("blur.rename", function () {
                $title.removeAttr("contenteditable");
                var new_title = $title.text().trim();
                if (new_title && new_title !== s.title) {
                    $.ajax({
                        url: "/api/chat/sessions/" + s.session_id + "/title",
                        method: "PATCH",
                        contentType: "application/json",
                        data: JSON.stringify({title: new_title}),
                    });
                    s.title = new_title;
                } else {
                    $title.text(s.title || "Untitled");
                }
            });
        });

        // Delete button
        item.find(".session-delete-btn").on("click", function (e) {
            e.stopPropagation();
            if (!confirm("Delete this chat?")) return;
            $.ajax({
                url: "/api/chat/sessions/" + s.session_id + "/delete",
                method: "DELETE",
                success: function () {
                    if (chat_state.session_id === s.session_id) {
                        $("#chat-messages").empty();
                        new_chat_session();
                        show_chat_system("Deleted. New session started.");
                    }
                    load_chat_sessions();
                },
            });
        });

        return item;
    }

    function load_chat_session(session_id) {
        $.getJSON("/api/chat/sessions/" + session_id, function (data) {
            chat_state.session_id = session_id;
            localStorage.setItem(CHAT_SESSION_KEY, session_id);
            var $msgs = $("#chat-messages");
            $msgs.empty();
            (data.messages || []).forEach(function (m) {
                if (m.role === "user" || m.role === "assistant") {
                    append_message(m.role, m.content);
                }
            });
            $msgs.scrollTop($msgs[0].scrollHeight);
            load_chat_sessions();  // refresh to highlight active
        });
    }

    // New chat button
    $("#btn-new-chat").on("click", function () {
        if (chat_state.streaming) return;
        $("#chat-messages").empty();
        new_chat_session();
        attached_files = [];
        render_attached_chips();
        show_chat_system("New chat started.");
        load_chat_sessions();
    });

    // Load sessions when chat tab opens
    $(".tab[data-tab='chat']").off("click.sessions").on("click.sessions", function () {
        load_chat_sessions();
    });

    // ── Expand to fullscreen modal ────────────────────────────────
    var _chat_in_modal = false;

    $("#btn-expand-chat").on("click", function () {
        if (_chat_in_modal) return;
        _chat_in_modal = true;
        var $body = $(".chat-body").detach();
        var $toolbar = $(".chat-toolbar").detach();
        var $modal_inner = $(".chat-modal-inner");
        $modal_inner.prepend($body).prepend($toolbar);
        $("#chat-modal").fadeIn(200);
    });

    $("#btn-collapse-chat").on("click", function () {
        if (!_chat_in_modal) return;
        _chat_in_modal = false;
        var $body = $(".chat-body").detach();
        var $toolbar = $(".chat-toolbar").detach();
        var $tab_chat = $("#tab-chat");
        $tab_chat.prepend($body).prepend($toolbar);
        $("#chat-modal").fadeOut(150);
    });

    // ── Attached files ────────────────────────────────────────────
    var attached_files = [];   // [{path, source, name}]

    function render_attached_chips() {
        var $area = $("#chat-attached-files");
        if (!attached_files.length) { $area.hide().empty(); return; }
        $area.show().empty();
        attached_files.forEach(function (f, i) {
            var chip =
                '<span class="attached-chip">' +
                    '<span class="attached-chip-icon">&#128196;</span>' +
                    escape_html(f.source + ":" + f.name) +
                    '<button class="attached-chip-remove" data-idx="' + i + '">&#x2715;</button>' +
                '</span>';
            $area.append(chip);
        });
    }

    $(document).on("click", ".attached-chip-remove", function () {
        var idx = parseInt($(this).data("idx"));
        attached_files.splice(idx, 1);
        render_attached_chips();
    });

    function attach_file(path, source) {
        var name = path.split("/").pop();
        // Avoid duplicates
        for (var i = 0; i < attached_files.length; i++) {
            if (attached_files[i].path === path && attached_files[i].source === source) return;
        }
        attached_files.push({path: path, source: source, name: name});
        render_attached_chips();
    }

    // ── Right panel toggle ────────────────────────────────────────
    var panel_open = false;

    $("#btn-toggle-right-panel").on("click", function () {
        panel_open = !panel_open;
        var $panel = $("#chat-right-panel");
        if (panel_open) {
            $panel.removeClass("collapsed");
            $(this).html("&#9664; Panel");
            if (panel_files_source === null) {
                init_file_browser();
            }
        } else {
            $panel.addClass("collapsed");
            $(this).html("&#9654; Panel");
        }
    });

    // ── Panel tabs ────────────────────────────────────────────────
    $(document).on("click", ".panel-tab", function () {
        var target = $(this).data("panel");
        $(".panel-tab").removeClass("active");
        $(this).addClass("active");
        $(".panel-section").removeClass("active");
        $("#panel-" + target).addClass("active");
    });

    // ── Terminal ──────────────────────────────────────────────────

    function term_log(html) {
        var $out = $("#terminal-output");
        $out.append('<div class="term-entry">' + html + '</div>');
        $out.scrollTop($out[0].scrollHeight);
    }

    function term_tool_use(name, query) {
        var icon = name === "search_memory" ? "&#128269;" :
                   name === "read_file"     ? "&#128196;" : "&#9881;";
        term_log(
            '<span class="term-tool">' + icon + ' ' + escape_html(name) + '</span>' +
            ' <span class="term-dim">' + escape_html(query) + '</span>'
        );
    }

    function term_bash_consent(data) {
        var id  = data.id;
        var cmd = data.command;
        var why = data.reason || "";
        var host = data.host || "local";
        var entry_id = "term-consent-" + id;
        term_log(
            '<div id="' + entry_id + '">' +
                '<div class="term-bash-header">&#9889; Bash request <span class="term-dim">(' + escape_html(host) + ')</span></div>' +
                '<pre class="term-cmd">' + escape_html(cmd) + '</pre>' +
                '<div class="term-dim">Reason: ' + escape_html(why) + '</div>' +
                '<div class="term-consent-btns">' +
                    '<button class="btn-consent-allow" data-cid="' + id + '">&#10003; Allow</button>' +
                    '<button class="btn-consent-deny"  data-cid="' + id + '">&#10007; Deny</button>' +
                '</div>' +
            '</div>'
        );
        // Auto-switch to terminal panel
        $(".panel-tab[data-panel='terminal']").trigger("click");
        if (!panel_open) { $("#btn-toggle-right-panel").trigger("click"); }
    }

    function term_bash_output(data) {
        var $entry = $("#term-consent-" + data.id);
        if ($entry.length) {
            $entry.find(".term-consent-btns").remove();
        }
        term_log('<pre class="term-output">' + escape_html(data.output) + '</pre>');
    }

    function term_bash_denied(data) {
        var $entry = $("#term-consent-" + data.id);
        if ($entry.length) $entry.find(".term-consent-btns").html('<span class="term-denied">&#10007; Denied</span>');
    }

    function term_bash_running(data) {
        var $entry = $("#term-consent-" + data.id);
        if ($entry.length) $entry.find(".term-consent-btns").html('<span class="term-running">&#8987; Running on ' + escape_html(data.host) + '…</span>');
    }

    // Consent button handlers
    $(document).on("click", ".btn-consent-allow", function () {
        var cid = $(this).data("cid");
        $(this).closest(".term-consent-btns").html('<span class="term-running">&#8987; Running…</span>');
        $.post("/api/bash/consent/" + cid + "?approved=true");
    });
    $(document).on("click", ".btn-consent-deny", function () {
        var cid = $(this).data("cid");
        $.post("/api/bash/consent/" + cid + "?approved=false");
        term_bash_denied({id: cid});
    });

    // Manual terminal input
    $("#btn-terminal-run").on("click", run_terminal_cmd);
    $("#terminal-cmd-input").on("keydown", function (e) {
        if (e.key === "Enter") { e.preventDefault(); run_terminal_cmd(); }
    });

    function run_terminal_cmd() {
        var cmd  = $("#terminal-cmd-input").val().trim();
        var host = $("#terminal-host-select").val();
        if (!cmd) return;
        $("#terminal-cmd-input").val("");
        term_log('<span class="term-prompt">' + escape_html(host) + ' $</span> <span class="term-cmd-inline">' + escape_html(cmd) + '</span>');

        $.ajax({
            url: "/api/bash/run",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({command: cmd, host: host}),
            success: function (r) {
                term_log('<pre class="term-output">' + escape_html(r.output || "") + '</pre>');
            },
            error: function (xhr) {
                term_log('<span class="term-error">Error: ' + escape_html(xhr.responseJSON?.detail || xhr.statusText) + '</span>');
            },
        });
    }

    // ── Handle new SSE events (tool_use, bash_consent, bash_output…) ─
    // Patch do_send to route terminal events from stream
    var _orig_handle_event = null;   // patched below in do_send

    // ── File browser ──────────────────────────────────────────────

    var panel_files_source = null;   // currently selected source id
    var panel_files_path   = "/";    // current path

    function init_file_browser() {
        $.getJSON("/api/files/sources", function (sources) {
            var $tabs = $("#file-source-tabs").empty();
            sources.forEach(function (s) {
                if (!s.available) return;
                var $btn = $('<button class="file-source-tab">')
                    .text(s.label)
                    .data("source", s.id)
                    .on("click", function () {
                        $(".file-source-tab").removeClass("active");
                        $(this).addClass("active");
                        panel_files_source = s.id;
                        panel_files_path   = "/";
                        load_file_listing("/");
                    });
                $tabs.append($btn);
            });
            // Auto-select first available
            $tabs.find(".file-source-tab").first().trigger("click");
        });
    }

    function load_file_listing(path) {
        panel_files_path = path;
        render_breadcrumb(path);
        $("#file-listing").html('<p class="empty-state" style="font-size:.8em">Loading…</p>');

        $.getJSON("/api/files/ls?path=" + encodeURIComponent(path) + "&source=" + encodeURIComponent(panel_files_source),
            function (entries) {
                var $list = $("#file-listing").empty();
                if (!entries.length) {
                    $list.html('<p class="empty-state" style="font-size:.8em">Empty directory.</p>');
                    return;
                }
                entries.forEach(function (e) {
                    var icon = e.type === "dir" ? "&#128193;" : "&#128196;";
                    var size_str = e.type === "file" ? fmt_size(e.size) : "";
                    var $row = $(
                        '<div class="file-row" data-type="' + e.type + '">' +
                            '<span class="file-icon">' + icon + '</span>' +
                            '<span class="file-name">' + escape_html(e.name) + '</span>' +
                            '<span class="file-meta">' + escape_html(size_str) + '</span>' +
                        '</div>'
                    );
                    $row.on("click", function () {
                        var full_path = (panel_files_path === "/" ? "" : panel_files_path) + "/" + e.name;
                        if (e.type === "dir") {
                            load_file_listing(full_path);
                        } else {
                            attach_file(full_path, panel_files_source);
                            show_chat_system("Attached: " + panel_files_source + ":" + full_path);
                        }
                    });
                    $list.append($row);
                });
            }
        ).fail(function (xhr) {
            $("#file-listing").html('<p class="empty-state" style="font-size:.8em;color:var(--danger)">Error: ' +
                escape_html(xhr.responseJSON?.detail || xhr.statusText) + '</p>');
        });
    }

    function render_breadcrumb(path) {
        var $bc  = $("#file-breadcrumb").empty();
        var parts = path.split("/").filter(Boolean);
        var $root = $('<span class="bc-part">').text("/").on("click", function () { load_file_listing("/"); });
        $bc.append($root);
        var accumulated = "";
        parts.forEach(function (p) {
            accumulated += "/" + p;
            var snap = accumulated;
            $bc.append($('<span class="bc-sep">').text(" / "));
            $bc.append($('<span class="bc-part">').text(p).on("click", function () { load_file_listing(snap); }));
        });
    }

    function fmt_size(bytes) {
        if (!bytes) return "";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }

    // ── Patch do_send to include attached files + route terminal events ──

    var _original_do_send = do_send;

    function do_send() {
        if (chat_state.streaming) return;
        var msg = $("#chat-input").val().trim();
        if (!msg) return;
        if (!chat_state.session_id) new_chat_session();

        $("#chat-input").val("").prop("disabled", true);
        $("#btn-chat-send").prop("disabled", true);
        chat_state.streaming = true;

        append_message("user", msg);

        var stream_id = "stream-" + Date.now();
        var html =
            '<div class="chat-msg chat-msg-assistant" id="' + stream_id + '">' +
                '<span class="chat-msg-label">Anamnesis</span>' +
                '<div class="chat-tool-uses"></div>' +
                '<div class="chat-msg-content"><span class="chat-cursor">&#9646;</span></div>' +
            "</div>";
        $("#chat-messages").append(html);
        scroll_chat();

        // Build payload with attached files
        var files_payload = attached_files.map(function (f) {
            return {path: f.path, source: f.source};
        });

        var payload = JSON.stringify({
            message:        msg,
            backend:        chat_state.backend,
            model:          chat_state.model || undefined,
            session_id:     chat_state.session_id,
            top_k:          5,
            attached_files: files_payload,
        });

        // Clear attached files after send
        attached_files = [];
        render_attached_chips();

        // ── Anamnesis retrieval: bypass LLM entirely ──────────────
        if (chat_state.backend === "anamnesis") {
            do_anamnesis_search(msg, stream_id);
            return;
        }

        fetch("/api/chat/stream", {
            method:  "POST",
            headers: {"Content-Type": "application/json"},
            body:    payload,
        }).then(function (resp) {
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            var reader  = resp.body.getReader();
            var decoder = new TextDecoder();
            var buffer  = "";
            var accumulated = "";

            function read_chunk() {
                reader.read().then(function (result) {
                    if (result.done) { finish_streaming(); return; }
                    buffer += decoder.decode(result.value, {stream: true});
                    var lines = buffer.split("\n");
                    buffer = lines.pop();

                    lines.forEach(function (line) {
                        if (!line.startsWith("data:")) return;
                        var raw = line.slice(5).trim();
                        if (!raw) return;
                        try {
                            var ev = JSON.parse(raw);

                            if (ev.token) {
                                accumulated += ev.token;
                                $("#" + stream_id + " .chat-msg-content")
                                    .html(escape_html(accumulated) + '<span class="chat-cursor">&#9646;</span>');
                                scroll_chat();

                            } else if (ev.tool_use) {
                                // Inline chip in message
                                var q = ev.tool_use.query || "";
                                var n = ev.tool_use.name  || "";
                                $("#" + stream_id + " .chat-tool-uses").append(
                                    '<div class="chat-tool-use-item">' +
                                        '<span class="chat-tool-use-icon">&#128269;</span> ' +
                                        escape_html(n) + ': <em>' + escape_html(q) + '</em>' +
                                    '</div>'
                                );
                                scroll_chat();
                                term_tool_use(n, q);

                            } else if (ev.bash_consent) {
                                term_bash_consent(ev.bash_consent);

                            } else if (ev.bash_running) {
                                term_bash_running(ev.bash_running);

                            } else if (ev.bash_output) {
                                term_bash_output(ev.bash_output);

                            } else if (ev.bash_denied || ev.bash_timeout) {
                                term_bash_denied(ev.bash_denied || ev.bash_timeout);

                            } else if (ev.error) {
                                $("#" + stream_id + " .chat-msg-content")
                                    .html('<span style="color:var(--danger)">' + escape_html(ev.error) + "</span>");
                                finish_streaming(); return;

                            } else if (ev.done) {
                                $("#" + stream_id + " .chat-msg-content")
                                    .text(accumulated).removeClass("chat-streaming");
                                finish_streaming(); return;
                            }
                        } catch(e) { /* ignore */ }
                    });

                    read_chunk();
                }).catch(function (err) {
                    $("#" + stream_id + " .chat-msg-content")
                        .html('<span style="color:var(--danger)">Stream error: ' + escape_html(err.message) + "</span>");
                    finish_streaming();
                });
            }
            read_chunk();

        }).catch(function (err) {
            $("#" + stream_id + " .chat-msg-content")
                .html('<span style="color:var(--danger)">Error: ' + escape_html(err.message) + "</span>");
            finish_streaming();
        });
    }

    // ── AnamnesisGPT generation (streaming) ────────────────────────
    function do_anamnesis_search(query, stream_id) {
        var $content = $("#" + stream_id + " .chat-msg-content");
        $content.addClass("chat-streaming").text("");
        var accumulated = "";

        fetch("/api/anamnesis-gpt/generate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({prompt: query, max_tokens: 512, temperature: 0.8, stream: true}),
        }).then(function (resp) {
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            var reader = resp.body.getReader();
            var decoder = new TextDecoder();
            var buffer = "";

            function read_chunk() {
                reader.read().then(function (result) {
                    if (result.done) {
                        $content.text(accumulated).removeClass("chat-streaming");
                        finish_streaming();
                        return;
                    }
                    buffer += decoder.decode(result.value, {stream: true});
                    var lines = buffer.split("\n");
                    buffer = lines.pop();
                    lines.forEach(function (line) {
                        if (!line.startsWith("data:")) return;
                        try {
                            var ev = JSON.parse(line.slice(5).trim());
                            if (ev.token) {
                                accumulated += ev.token;
                                $content.text(accumulated);
                                scroll_chat();
                            }
                            if (ev.done) {
                                $content.text(accumulated).removeClass("chat-streaming");
                                finish_streaming();
                                return;
                            }
                        } catch(e) { /* ignore */ }
                    });
                    read_chunk();
                }).catch(function (err) {
                    $content.html('<span style="color:var(--danger)">Stream error: ' + escape_html(err.message) + "</span>");
                    finish_streaming();
                });
            }
            read_chunk();
        }).catch(function (err) {
            $content.html('<span style="color:var(--danger)">Error: ' + escape_html(err.message) + "</span>");
            finish_streaming();
        });
    }

    // Re-bind send buttons to patched do_send
    $("#btn-chat-send").off("click").on("click", do_send);
    $("#chat-input").off("keydown").on("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); do_send(); }
    });

    // ── Embedding Tab ─────────────────────────────────────────────

    var embed_config = {};

    function load_embedding_config() {
        $.getJSON("/api/embedding/config", function (data) {
            embed_config = data;
            $("#embed-active-model").text(data.model_id || "—");
            $("#embed-active-dims").text(data.dimensions || "—");
            $("#embed-active-workers").text(data.pool_workers || "—");

            // Model selector
            var sel = $("#embed-model-select").empty();
            (data.available_models || []).forEach(function (m) {
                var opt = $("<option>").val(m.model_id).text(
                    m.display_name + " (" + m.dimensions + " dims)"
                );
                if (m.model_id === data.model_id) opt.prop("selected", true);
                sel.append(opt);
            });
            update_embed_model_hint();

            // Core checkboxes — mark currently active cores (first pool_workers cores)
            var box = $("#embed-core-checkboxes").empty();
            var n_active = data.pool_workers || 0;
            for (var c = 0; c < (data.total_cores || 1); c++) {
                var checked = c < n_active;
                box.append(
                    $("<label>").css({ display: "flex", alignItems: "center", gap: "3px", cursor: "pointer" }).append(
                        $("<input>").attr({ type: "checkbox", "data-core": c }).prop("checked", checked),
                        $("<span>").text(c)
                    )
                );
            }
        });
    }

    function update_embed_model_hint() {
        var sel_id = $("#embed-model-select").val();
        var models = embed_config.available_models || [];
        var info = models.find(function (m) { return m.model_id === sel_id; });
        $("#embed-model-hint").text(info ? info.notes : "");
    }

    $("#embed-model-select").on("change", update_embed_model_hint);

    $("#btn-embed-apply-model").on("click", function () {
        var model_id = $("#embed-model-select").val();
        if (!model_id) return;
        if (!confirm("Switch to " + model_id + "?\n\nAll episodes will be re-embedded automatically.")) return;
        var btn = this;
        $(btn).prop("disabled", true).text("Loading…");
        $.ajax({
            url: "/api/embedding/config",
            method: "PUT",
            contentType: "application/json",
            data: JSON.stringify({ model_id: model_id }),
            success: function (data) {
                load_embedding_config();
                if (data.reembed === "started") {
                    reembed_poll = setInterval(poll_reembed_status, 2000);
                    poll_reembed_status();
                }
            },
            error: function (xhr) { alert("Model reload failed: " + xhr.responseText); },
            complete: function () { $(btn).prop("disabled", false).text("Apply Model"); }
        });
    });

    function selected_cores() {
        var cores = [];
        $("#embed-core-checkboxes input[type=checkbox]").each(function () {
            if ($(this).prop("checked")) cores.push(parseInt($(this).attr("data-core"), 10));
        });
        return cores;
    }

    function select_first_pct(pct) {
        var total = embed_config.total_cores || 1;
        var n = Math.max(1, Math.round(total * pct / 100));
        $("#embed-core-checkboxes input[type=checkbox]").each(function (i) {
            $(this).prop("checked", i < n);
        });
    }

    $("#btn-cores-all").on("click", function () { select_first_pct(100); });
    $("#btn-cores-half").on("click", function () { select_first_pct(50); });
    $("#btn-cores-quarter").on("click", function () { select_first_pct(25); });

    $("#btn-embed-apply-cores").on("click", function () {
        var cores = selected_cores();
        if (cores.length === 0) { alert("Select at least one core."); return; }
        var btn = this;
        $(btn).prop("disabled", true).text("Applying…");
        $.ajax({
            url: "/api/embedding/config",
            method: "PUT",
            contentType: "application/json",
            data: JSON.stringify({ cpu_cores: cores }),
            success: function () { load_embedding_config(); },
            error: function (xhr) { alert("Failed: " + xhr.responseText); },
            complete: function () { $(btn).prop("disabled", false).text("Apply Affinity"); }
        });
    });

    // ─── Reembed ──────────────────────────────────────────────────
    var reembed_poll = null;

    function poll_reembed_status() {
        $.getJSON("/api/episodes/reembed/status", function (data) {
            var msg;
            if (data.running) {
                var pct = data.total ? Math.round(data.done / data.total * 100) : 0;
                msg = "Re-embedding… " + data.done + " / " + data.total + " (" + pct + "%)" +
                      (data.errors ? " &nbsp;<span style='color:var(--danger)'>(" + data.errors + " errors)</span>" : "");
            } else if (data.paused && data.checkpoint) {
                msg = "Paused at " + data.done + " / " + data.total + " — checkpoint saved. Click Resume to continue.";
            } else if (data.stale) {
                msg = "Embeddings stale — re-embed needed.";
            } else if (data.current_model) {
                msg = "Up to date &nbsp;(" + data.current_model + ", " + data.done + " episodes)";
            } else if (data.checkpoint) {
                msg = "Checkpoint found at " + data.checkpoint.done + " / " + data.checkpoint.total + " (" + data.checkpoint.model_id + "). Click Resume.";
            } else {
                msg = "Not yet run this session.";
            }
            $("#reembed-status").show().html(msg);

            // Buttons
            var has_checkpoint = !!(data.paused || data.checkpoint);
            var up_to_date = !data.stale && !data.running && !has_checkpoint && data.done > 0;

            $("#btn-reembed")
                .prop("disabled", up_to_date || data.running)
                .css("opacity", (up_to_date || data.running) ? 0.4 : 1)
                .text(has_checkpoint && !data.running ? "Re-embed (fresh start)" : "Re-embed All");

            $("#btn-reembed-pause")
                .toggle(!!data.running)
                .text(data.pause_requested ? "Pausing…" : "Pause");

            $("#btn-reembed-resume")
                .toggle(!data.running && has_checkpoint);

            if (!data.running && reembed_poll) {
                clearInterval(reembed_poll);
                reembed_poll = null;
            }
        });
    }

    function start_reembed_poll() {
        if (!reembed_poll) reembed_poll = setInterval(poll_reembed_status, 2000);
        poll_reembed_status();
    }

    $("#btn-reembed").on("click", function () {
        if (!confirm("Re-embed ALL episodes with the current model?\nThis may take a long time on CPU.")) return;
        $(this).prop("disabled", true);
        $.ajax({
            url: "/api/episodes/reembed",
            method: "POST",
            success: start_reembed_poll,
            error: function (xhr) { alert("Failed: " + xhr.responseText); },
            complete: function () { $("#btn-reembed").prop("disabled", false); }
        });
    });

    $("#btn-reembed-pause").on("click", function () {
        $(this).prop("disabled", true).text("Pausing…");
        $.ajax({
            url: "/api/episodes/reembed/pause",
            method: "POST",
            complete: function () {
                $(this).prop("disabled", false);
                poll_reembed_status();
            }
        });
    });

    $("#btn-reembed-resume").on("click", function () {
        $(this).prop("disabled", true);
        $.ajax({
            url: "/api/episodes/reembed/resume",
            method: "POST",
            success: start_reembed_poll,
            error: function (xhr) { alert("Failed: " + xhr.responseText); },
            complete: function () { $(this).prop("disabled", false); }
        });
    });

    $(".tab[data-tab='embedding']").on("click", function () {
        load_embedding_config();
        poll_reembed_status();
    });

    // ─── Architecture diagram toggle ─────────────────────────────
    var _arch_peek_done = false;

    $(".tab[data-tab='architecture']").on("click", function () {
        if (_arch_peek_done) return;
        _arch_peek_done = true;
        var wrap = $("#arch-diagram-wrap");
        var toggle = $("#arch-diagram-toggle");
        if (typeof maybeRenderMermaid === "function") maybeRenderMermaid();
        wrap.slideDown(900, function () {
            setTimeout(function () {
                wrap.slideUp(250);
                toggle.removeClass("open");
            }, 600);
        });
    });

    $("#arch-diagram-toggle").on("click", function () {
        var wrap = $("#arch-diagram-wrap");
        var toggle = $(this);
        if (wrap.is(":visible")) {
            wrap.slideUp(200);
            toggle.removeClass("open");
        } else {
            if (typeof maybeRenderMermaid === "function") maybeRenderMermaid();
            wrap.slideDown(800);
            toggle.addClass("open");
        }
    });

});

/* ── Training Tab ────────────────────────────────────────────────── */

(function () {

    var TRAINERS = [];   // populated from /api/config/trainers

    const cards = {};   // name → { $el, chart, history }

    function drawSparkline(canvas, history) {
        if (!canvas || !history.length) return;
        const ctx = canvas.getContext("2d");
        const W = canvas.offsetWidth || 400;
        const H = canvas.height;
        canvas.width = W;
        ctx.clearRect(0, 0, W, H);

        const losses = history.map(h => h.loss);
        const min = Math.min(...losses), max = Math.max(...losses) || 1;
        const pad = 4;

        ctx.beginPath();
        ctx.strokeStyle = "#58a6ff";
        ctx.lineWidth = 1.5;
        losses.forEach((v, i) => {
            const x = pad + (i / (losses.length - 1 || 1)) * (W - pad * 2);
            const y = H - pad - ((v - min) / (max - min || 1)) * (H - pad * 2);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    function statusDot(running, done, err) {
        if (err)     return "error";
        if (running) return "running";
        if (done)    return "done";
        return "";
    }

    function statusLabel(running, done, err) {
        if (err)     return "unreachable";
        if (running) return "training";
        if (done)    return "done";
        return "idle";
    }

    function updateCard(name, data, err) {
        const c = cards[name];
        if (!c) return;
        const $el = c.$el;

        const running = !err && data && data.running;
        const done    = !err && data && data.done;
        const cls = statusDot(running, done, err);

        $el.find(".trainer-status-dot").attr("class", "trainer-status-dot " + cls);
        $el.find(".trainer-status-label").text(statusLabel(running, done, err));

        const p = data && data.progress;
        const pct = p ? p.pct : 0;
        $el.find(".trainer-progress-bar").css("width", pct + "%");
        $el.find(".trainer-progress-label").text(
            p ? `step ${p.step}/${p.total} (${pct}%) · ETA ${p.eta}` : "—"
        );

        const m = data && data.latest_metrics;
        $el.find(".tk-loss").text(m ? m.loss.toFixed(3) : "—");
        $el.find(".tk-acc").text(m ? (m.accuracy * 100).toFixed(1) + "%" : "—");
        $el.find(".tk-epoch").text(m ? m.epoch.toFixed(3) : "—");
        $el.find(".tk-eta").text(p ? p.eta : "—");
        $el.find(".tk-sps").text(p ? p.sec_per_step.toFixed(1) + "s" : "—");

        const g = data && data.gpu;
        $el.find(".tk-gpu-pct").text(g && g.gpu_pct != null ? g.gpu_pct.toFixed(0) + "%" : "—");
        $el.find(".tk-vram").text(g && g.vram_used_mb != null
            ? (g.vram_used_mb / 1024).toFixed(1) + "/" + (g.vram_total_mb / 1024).toFixed(1) + "G"
            : "—");
        $el.find(".tk-temp").text(g && g.temp_c != null ? g.temp_c.toFixed(0) + "°C" : "—");
        $el.find(".tk-power").text(g && g.power_w != null ? g.power_w.toFixed(0) + "W" : "—");

        // Sparkline
        if (data && data.history && data.history.length) {
            c.history = data.history;
            drawSparkline($el.find(".trainer-loss-chart")[0], c.history);
        }
    }

    function pollTrainer(t) {
        $.ajax({
            url: t.url + "/status",
            timeout: 6000,
            success: function (data) { updateCard(t.name, data, false); },
            error:   function ()     { updateCard(t.name, null,  true);  },
        });
    }

    function buildCards() {
        const $container = $("#training-machines");
        $container.empty();
        const tpl = document.getElementById("tpl-trainer-card");

        TRAINERS.forEach(function (t) {
            const $clone = $(tpl.content.cloneNode(true));
            $clone.find(".trainer-name").text(t.name);
            $clone.find(".trainer-gpu-badge").text(t.label || t.name);
            const $card = $clone.find(".trainer-card");

            // Start button
            $clone.find(".btn-trainer-start").on("click", function () {
                $.post(t.url + "/start", JSON.stringify({}), null, "json")
                    .always(function () { pollTrainer(t); });
            });

            // Stop button
            $clone.find(".btn-trainer-stop").on("click", function () {
                $.post(t.url + "/stop")
                    .always(function () { pollTrainer(t); });
            });

            // Log tail toggle
            $clone.find(".btn-trainer-log").on("click", function () {
                const $log = $(this).siblings(".trainer-log-output");
                if ($log.is(":visible")) {
                    $log.hide();
                    $(this).text("Show Log");
                } else {
                    $.getJSON(t.url + "/log/tail?lines=60", function (d) {
                        $log.text((d.lines || []).join("\n")).show();
                    });
                    $(this).text("Hide Log");
                }
            });

            $container.append($clone);
            cards[t.name] = { $el: $container.find(".trainer-card").last(), history: [] };
            pollTrainer(t);
        });
    }

    // Poll every 10s when tab is active
    var _pollInterval = null;

    $(".tab[data-tab='training']").on("click", function () {
        $.getJSON("/api/config/trainers", function (cfg) {
            TRAINERS = cfg.trainers || [];
            buildCards();
            clearInterval(_pollInterval);
            _pollInterval = setInterval(function () {
                TRAINERS.forEach(pollTrainer);
            }, 10000);
        });
    });

    // Stop polling when leaving training tab
    $(".tab:not([data-tab='training'])").on("click", function () {
        clearInterval(_pollInterval);
    });

})();

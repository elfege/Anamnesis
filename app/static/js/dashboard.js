/* ─── Anamnesis Dashboard JS (jQuery) ────────────────────────── */

$(function () {

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

    // ── Claude toggle ─────────────────────────────────────────────
    $("#btn-claude-toggle").on("click", function () {
        if (chat_state.claude_enabled) {
            disable_claude();
        } else {
            enable_claude();
        }
    });

    function enable_claude() {
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
        var label = role === "user" ? "You" : (chat_state.backend === "claude" ? "Claude" : "Ollama");
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
        var label = chat_state.backend === "claude" ? "Claude" : "Ollama";
        var html =
            '<div class="chat-msg chat-msg-assistant" id="' + stream_id + '">' +
                '<span class="chat-msg-label">' + label + '</span>' +
                '<div class="chat-msg-content chat-streaming">&#9646;</div>' +
            "</div>";
        $("#chat-messages").append(html);
        scroll_chat();

        var payload = JSON.stringify({
            message: msg,
            backend: chat_state.backend,
            model: chat_state.model || undefined,
            session_id: chat_state.session_id,
            top_k: 3,
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
                                    .text(accumulated + "▌");
                                scroll_chat();
                            } else if (event.error) {
                                $("#" + stream_id + " .chat-msg-content")
                                    .html('<span style="color:var(--danger)">' + escape_html(event.error) + "</span>");
                                finish_streaming();
                                return;
                            } else if (event.done) {
                                $("#" + stream_id + " .chat-msg-content").text(accumulated);
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
});

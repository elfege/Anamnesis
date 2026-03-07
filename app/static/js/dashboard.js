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
});

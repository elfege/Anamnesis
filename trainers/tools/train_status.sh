#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# train_status.sh — Visualize Anamnesis trainer progress
#
# Usage:
#   train_status                        # interactive menu
#   train_status --url http://host:3011 # direct, skip menu
#   train_status --machine server       # by alias
#   train_status --all                  # all machines at once
#   train_status --watch                # auto-refresh every 10s
#   train_status --watch 5              # auto-refresh every 5s
#   train_status --json                 # raw JSON output
#
# Source from .bash_aliases:
#   alias ts='bash ~/0_GENESIS_PROJECT/0_ANAMNESIS/trainers/tools/train_status.sh'
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults & known machines ─────────────────────────────────────
declare -A MACHINES=(
	[server]="http://192.168.10.15:3011"
	[office]="http://192.168.10.110:3011"
)

URL=""
WATCH=0
WATCH_INTERVAL=10
JSON_MODE=0
ALL_MODE=0

# ── Colors ────────────────────────────────────────────────────────
R='\033[0;31m' G='\033[0;32m' Y='\033[0;33m' C='\033[0;36m'
B='\033[1m' D='\033[2m' NC='\033[0m'
BG_G='\033[42;30m' BG_R='\033[41;37m' BG_Y='\033[43;30m' BG_C='\033[46;30m'

# ── Arg parsing ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
	case "$1" in
	--url | -u)
		URL="$2"
		shift 2
		;;
	--machine | -m)
		key="${2,,}" # lowercase
		if [[ -n "${MACHINES[$key]+_}" ]]; then
			URL="${MACHINES[$key]}"
		else
			echo -e "${R}Unknown machine: $2${NC}"
			echo "Known: ${!MACHINES[*]}"
			exit 1
		fi
		shift 2
		;;
	--all | -a)
		ALL_MODE=1
		shift
		;;
	--watch | -w)
		WATCH=1
		if [[ ${2:-} =~ ^[0-9]+$ ]]; then
			WATCH_INTERVAL="$2"
			shift
		fi
		shift
		;;
	--json | -j)
		JSON_MODE=1
		shift
		;;
	--help | -h)
		head -16 "$0" | tail -13
		exit 0
		;;
	*)
		echo -e "${R}Unknown arg: $1${NC}"
		exit 1
		;;
	esac
done

# ── Interactive menu (if no URL and not --all) ────────────────────
if [[ -z "$URL" && $ALL_MODE -eq 0 ]]; then
	echo -e "${C}${B}Anamnesis Trainer Status${NC}"
	echo ""
	i=1
	menu_keys=()
	for name in $(echo "${!MACHINES[@]}" | tr ' ' '\n' | sort); do
		echo -e "  ${B}$i)${NC} $name  ${D}${MACHINES[$name]}${NC}"
		menu_keys+=("$name")
		((i++))
	done
	echo -e "  ${B}$i)${NC} All machines"
	echo ""
	read -rp "Select [1-$i]: " choice

	if [[ "$choice" == "$i" ]]; then
		ALL_MODE=1
	elif [[ "$choice" =~ ^[0-9]+$ ]] && ((choice >= 1 && choice < i)); then
		URL="${MACHINES[${menu_keys[$((choice - 1))]}]}"
	else
		echo -e "${R}Invalid choice${NC}"
		exit 1
	fi
fi

# ── Fetch & render ────────────────────────────────────────────────

fetch_status() {
	local url="$1"
	curl -sf --max-time 5 "$url/status" 2>/dev/null
}

bar() {
	# bar <pct> <width> <fill_color>
	local pct=${1:-0} width=${2:-40} color=${3:-$G}
	local filled=$((pct * width / 100))
	local empty=$((width - filled))
	printf "${color}"
	printf '█%.0s' $(seq 1 $((filled > 0 ? filled : 1))) 2>/dev/null || printf '░'
	printf "${D}"
	printf '░%.0s' $(seq 1 $((empty > 0 ? empty : 1))) 2>/dev/null || true
	printf "${NC}"
}

render_machine() {
	local name="$1" url="$2"
	local data
	data=$(fetch_status "$url" 2>/dev/null) || {
		echo -e "  ${BG_R} $name ${NC} ${R}unreachable${NC}  ${D}$url${NC}"
		echo ""
		return
	}

	if [[ $JSON_MODE -eq 1 ]]; then
		echo "$data" | python3 -m json.tool
		return
	fi

	# Parse with python (reliable JSON handling)
	eval "$(echo "$data" | python3 -c '
import json, sys
s = json.load(sys.stdin)
p = s.get("progress", {})
m = s.get("latest_metrics", {})
g = s.get("gpu", {})
h = s.get("history", [])
print(f"running={int(s.get(\"running\", False))}")
print(f"done={int(s.get(\"done\", False))}")
print(f"exit_code={s.get(\"exit_code\", \"null\")}")
print(f"pid={s.get(\"pid\", \"?\")}")
print(f"started_at={s.get(\"started_at\", \"?\")}")
print(f"pct={p.get(\"pct\", 0)}")
print(f"step={p.get(\"step\", 0)}")
print(f"total={p.get(\"total\", 0)}")
print(f"elapsed={p.get(\"elapsed\", \"--\")}")
print(f"eta={p.get(\"eta\", \"--\")}")
print(f"sec_per_step={p.get(\"sec_per_step\", 0)}")
print(f"loss={m.get(\"loss\", \"--\")}")
print(f"lr={m.get(\"lr\", \"--\")}")
print(f"accuracy={m.get(\"accuracy\", \"--\")}")
print(f"epoch={m.get(\"epoch\", \"--\")}")
print(f"gpu_pct={g.get(\"gpu_pct\", 0)}")
print(f"vram_used={g.get(\"vram_used_mb\", 0)}")
print(f"vram_total={g.get(\"vram_total_mb\", 0)}")
print(f"temp={g.get(\"temp_c\", 0)}")
print(f"power={g.get(\"power_w\", 0)}")
print(f"gpu_type={s.get(\"gpu_type\", \"?\")}")
# Loss history sparkline (last 20 points)
losses = [x["loss"] for x in h[-20:] if "loss" in x]
print(f"loss_hist={\" \".join(str(l) for l in losses)}")
' 2>/dev/null)" || {
		echo -e "  ${BG_R} $name ${NC} ${R}parse error${NC}"
		return
	}

	# ── Header ──
	if [[ $running -eq 1 ]]; then
		echo -e "  ${BG_G} $name ${NC}  ${G}TRAINING${NC}  ${D}pid:$pid  $url${NC}"
	elif [[ $done -eq 1 && "$exit_code" == "0" ]]; then
		echo -e "  ${BG_G} $name ${NC}  ${G}COMPLETE${NC}  ${D}$url${NC}"
	elif [[ $done -eq 1 ]]; then
		echo -e "  ${BG_R} $name ${NC}  ${R}FAILED (exit $exit_code)${NC}  ${D}$url${NC}"
	else
		echo -e "  ${BG_Y} $name ${NC}  ${Y}IDLE${NC}  ${D}$url${NC}"
	fi

	# ── Progress bar ──
	if [[ $total -gt 0 ]]; then
		printf "  "
		bar "$pct" 40 "$C"
		echo -e "  ${B}${pct}%%${NC}  step ${step}/${total}"
		echo -e "  ${D}elapsed: ${elapsed}  eta: ${eta}  ${sec_per_step}s/step${NC}"
	fi

	# ── Metrics ──
	if [[ "$loss" != "--" ]]; then
		echo -e "  ${B}loss:${NC} $loss  ${B}acc:${NC} $accuracy  ${B}lr:${NC} $lr  ${B}epoch:${NC} $epoch"
	fi

	# ── Loss sparkline ──
	if [[ -n "$loss_hist" ]]; then
		sparkline=$(echo "$loss_hist" | python3 -c '
import sys
vals = [float(x) for x in sys.stdin.read().split()]
if len(vals) < 2:
    sys.exit()
blocks = " ▁▂▃▄▅▆▇█"
mn, mx = min(vals), max(vals)
rng = mx - mn if mx != mn else 1
out = ""
for v in vals:
    idx = int((v - mn) / rng * 7)
    out += blocks[idx]
print(out)
' 2>/dev/null)
		if [[ -n "$sparkline" ]]; then
			echo -e "  ${D}loss trend:${NC} $sparkline  ${D}(${#loss_hist// /,} pts)${NC}"
		fi
	fi

	# ── GPU ──
	local vram_pct=0
	if [[ ${vram_total%.*} -gt 0 ]]; then
		vram_pct=$(python3 -c "print(int(${vram_used}/${vram_total}*100))" 2>/dev/null || echo 0)
	fi

	# Temp color
	local tc="$G"
	if ((${temp%.*} >= 80)); then tc="$R"
	elif ((${temp%.*} >= 65)); then tc="$Y"
	fi

	echo -ne "  ${B}GPU:${NC} ${gpu_pct}%%  "
	printf "${B}VRAM:${NC} "
	bar "$vram_pct" 20 "$Y"
	echo -e "  ${vram_used%.*}/${vram_total%.*} MB"
	echo -e "  ${B}Temp:${NC} ${tc}${temp}°C${NC}  ${B}Power:${NC} ${power}W  ${D}(${gpu_type})${NC}"
	echo ""
}

do_render() {
	if [[ $ALL_MODE -eq 1 ]]; then
		echo -e "${C}${B}═══ Anamnesis Training Status ═══${NC}  ${D}$(date '+%H:%M:%S')${NC}"
		echo ""
		for name in $(echo "${!MACHINES[@]}" | tr ' ' '\n' | sort); do
			render_machine "$name" "${MACHINES[$name]}"
		done
	else
		# Find machine name for this URL
		local mname="unknown"
		for name in "${!MACHINES[@]}"; do
			if [[ "${MACHINES[$name]}" == "$URL" ]]; then
				mname="$name"
				break
			fi
		done
		echo -e "${C}${B}═══ Anamnesis Training Status ═══${NC}  ${D}$(date '+%H:%M:%S')${NC}"
		echo ""
		render_machine "$mname" "$URL"
	fi
}

# ── Main loop ─────────────────────────────────────────────────────
if [[ $WATCH -eq 1 ]]; then
	while true; do
		clear
		do_render
		echo -e "${D}Refreshing every ${WATCH_INTERVAL}s — Ctrl+C to stop${NC}"
		sleep "$WATCH_INTERVAL"
	done
else
	do_render
fi

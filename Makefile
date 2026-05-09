
bootstrap :
	bash ./test.sh

NIX_FLAKES := flake.nix
SH_SCRIPTS := $(shell find . -name "*.sh")

# Transform flake.nix paths to various result paths
TEST_RESULTS := $(patsubst %.nix,%.test-result.json,$(NIX_FLAKES))
RAW_DUMPS := $(patsubst %.nix,%.raw-dump.txt,$(NIX_FLAKES))
JSON_DUMPS := $(patsubst %.nix,%.json-dump.json,$(NIX_FLAKES))
LINT_RESULTS := $(patsubst %.nix,%.lint-result.txt,$(NIX_FLAKES))
EVAL_RESULTS := $(patsubst %.nix,%.eval-result.txt,$(NIX_FLAKES))
TWOGRAM_REPORTS := $(patsubst %.nix,%.2gram-report.txt,$(NIX_FLAKES))

# Transform .sh paths to .shellcheck-result.txt paths
SHELLCHECK_RESULTS := $(patsubst %.sh,%.shellcheck-result.txt,$(SH_SCRIPTS))

VERBOSE_LOGS := $(patsubst %.nix,%.verbose-log.txt,$(NIX_FLAKES))

.PHONY: all clean help

all: $(TEST_RESULTS) $(SHELLCHECK_RESULTS) $(RAW_DUMPS) $(JSON_DUMPS) $(LINT_RESULTS) $(EVAL_RESULTS) $(TWOGRAM_REPORTS) $(VERBOSE_LOGS)

%.verbose-log.txt: %.nix
	@bash /data/data/com.termux.nix/files/home/pick-up-nix2/source/github/meta-introspector/ai-ml-zk-ops/flakes/repo-data-flake/scripts/dump_flake_verbose_log.sh $< $@


all: $(TEST_RESULTS) $(SHELLCHECK_RESULTS) $(RAW_DUMPS) $(JSON_DUMPS) $(LINT_RESULTS) $(EVAL_RESULTS) $(TWOGRAM_REPORTS)

help:
	@echo "For help with debugging flakes, see docs/tutorials/Debugging_Nix_Flakes.md"


%.test-result.json: %.nix
	@bash /data/data/com.termux.nix/files/home/pick-up-nix2/source/github/meta-introspector/ai-ml-zk-ops/flakes/repo-data-flake/scripts/build_test_result.sh $< $@

%.shellcheck-result.txt: %.sh
	@bash /data/data/com.termux.nix/files/home/pick-up-nix2/source/github/meta-introspector/ai-ml-zk-ops/flakes/repo-data-flake/scripts/run_shellcheck.sh $< $@

%.raw-dump.txt: %.nix
	@bash /data/data/com.termux.nix/files/home/pick-up-nix2/source/github/meta-introspector/ai-ml-zk-ops/flakes/repo-data-flake/scripts/dump_flake_raw.sh $< $@

%.json-dump.json: %.nix
	@bash /data/data/com.termux.nix/files/home/pick-up-nix2/source/github/meta-introspector/ai-ml-zk-ops/flakes/repo-data-flake/scripts/dump_flake_json.sh $< $@

%.lint-result.txt: %.nix
	@bash /data/data/com.termux.nix/files/home/pick-up-nix2/source/github/meta-introspector/ai-ml-zk-ops/flakes/repo-data-flake/scripts/lint_flake.sh $< $@

%.eval-result.txt: %.nix
	@bash /data/data/com.termux.nix/files/home/pick-up-nix2/source/github/meta-introspector/ai-ml-zk-ops/flakes/repo-data-flake/scripts/eval_flake.sh $< $@

%.2gram-report.txt: %.nix
	@bash /data/data/com.termux.nix/files/home/pick-up-nix2/source/github/meta-introspector/ai-ml-zk-ops/flakes/repo-data-flake/scripts/generate_2gram_report.sh $< $@

clean:
	@echo "Cleaning up all generated files..."
	$(RM) $(TEST_RESULTS) $(SHELLCHECK_RESULTS) $(RAW_DUMPS) $(JSON_DUMPS) $(LINT_RESULTS) $(EVAL_RESULTS) $(TWOGRAM_REPORTS)

# pkg-visualizer container targets.
#
# Container runtime: RUNTIME=docker (default) or RUNTIME=apple. RUNTIME=apple
# drives Apple's native `container` CLI (Apple Silicon, macOS 26, CLI 1.1+;
# no Docker Desktop). The image is multi-arch, so both run natively on arm64.
#
#   make build                 -- build the image
#   make gui                   -- GUI on http://localhost:6080 (noVNC)
#   make export                -- headless HTML export into $(OUT)
#   make export ARGS="--full -s /out/mypkg.png"   -- any pkg-visualizer options
#   make logs / make stop      -- follow logs / remove the GUI container
#   make push                  -- multi-arch build + push to Docker Hub (Docker only)
#
# Pick the package with PKG=/path/to/package (default: this repo's source).
# Apple containers are one VM each; size it with MEM=4g (default 2g).
#   make gui RUNTIME=apple PKG=~/repos/turtlend/src/turtlend

RUNTIME ?= docker
IMAGE   ?= egsuchanek/pkg-visualizer
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
NAME    ?= pkg-visualizer
PORT    ?= 6080
PKG     ?= $(CURDIR)/pkg_visualizer
PKG_NAME ?= $(notdir $(abspath $(PKG)))
OUT     ?= $(CURDIR)/out
MEM     ?= 2g
ARGS    ?=

ifeq ($(RUNTIME),apple)
CLI      := container
RUN_OPTS := --memory $(MEM)
else ifeq ($(RUNTIME),docker)
CLI      := docker
RUN_OPTS :=
else
$(error RUNTIME must be docker or apple, got '$(RUNTIME)')
endif

MOUNTS = -e PKG_NAME=$(PKG_NAME) -v $(abspath $(PKG)):/pkg:ro -v $(OUT):/out

.PHONY: setup build gui export logs stop push

setup:
ifeq ($(RUNTIME),apple)
	@command -v container >/dev/null || { echo "Install Apple's container CLI: brew install container"; exit 1; }
	@container system status >/dev/null 2>&1 || container system start --enable-kernel-install
endif

build: setup
	$(CLI) build -f Dockerfile -t $(IMAGE):latest -t $(IMAGE):$(VERSION) .

gui: setup
	@mkdir -p $(OUT)
	-@$(CLI) rm -f $(NAME) >/dev/null 2>&1
	$(CLI) run -d --name $(NAME) $(RUN_OPTS) -p 127.0.0.1:$(PORT):6080 $(MOUNTS) $(IMAGE):latest $(ARGS)
	@echo "pkg-visualizer GUI: http://localhost:$(PORT)/vnc.html?autoconnect=1&resize=scale"

export: setup
	@mkdir -p $(OUT)
	$(CLI) run --rm $(RUN_OPTS) $(MOUNTS) $(IMAGE):latest --headless $(ARGS)

logs:
	$(CLI) logs -f $(NAME)

stop:
	-$(CLI) rm -f $(NAME)

push:
ifeq ($(RUNTIME),apple)
	$(error make push uses docker buildx for the multi-arch manifest; run it with RUNTIME=docker)
endif
	docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile \
	  -t $(IMAGE):latest -t $(IMAGE):$(VERSION) --push .

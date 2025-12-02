.PHONY: build-linux build-linux-arm64 release-linux release-linux-arm64

build:
	ORT_SKIP_DOWNLOAD=1 cross build --target aarch64-unknown-linux-gnu

release:
	ORT_SKIP_DOWNLOAD=1 cross build --target aarch64-unknown-linux-gnu

clippy :
	ORT_SKIP_DOWNLOAD=1 cross clippy --target aarch64-unknown-linux-gnu

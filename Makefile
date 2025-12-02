.PHONY: build-linux build-linux-arm64 release-linux release-linux-arm64

build-linux:
	ORT_SKIP_DOWNLOAD=1 cross build --target x86_64-unknown-linux-gnu

build-linux-arm64:
	ORT_SKIP_DOWNLOAD=1 cross build --target aarch64-unknown-linux-gnu

release-linux:
	ORT_SKIP_DOWNLOAD=1 cross build --release --target x86_64-unknown-linux-gnu

release-linux-arm64:
	ORT_SKIP_DOWNLOAD=1 cross build --release --target aarch64-unknown-linux-gnu

SHELL := /bin/bash
.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean: ## Clean the project
	cargo clean
	rm -f $(TARGET)

fmt: ## Format the code
	@rustup component add rustfmt
	cargo fmt

build: ## Build the project
	cargo build --bin report --features="cli"

run: ## Run the project
	cargo run -- --num1 1.2 --num2 4.1 --output output

test: ## Run the tests
	cargo test

lint: ## Run the linter
	@rustup component add clippy
	cargo clippy

doc: ## Generate the documentation
	cargo doc --no-deps --document-private-items

example: ## Run the fetch example
	cargo run --example fetch

TARGET := target/release/report
release: ## Build the project in release mode
	cargo build --bin report --release --features="cli"
	@echo "Binary is located at $(TARGET)"


{
  description = "Nixified Rust project: llm";

  inputs = {
    nixpkgs.url = "github:meta-introspector/nixpkgs?ref=feature/CRQ-016-nixify";
    naersk.url = "github:meta-introspector/naersk?ref=feature/CRQ-016-nixify";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, naersk, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        naersk-lib = naersk.lib.${system};
      in
      {
        packages.default = naersk-lib.buildPackage {
          pname = "llm";
          version = "1.3.5"; # Get this from Cargo.toml
          src = ./.;
          nativeBuildInputs = with pkgs; [
            pkg-config
            openssl
          ];
          buildInputs = with pkgs; [
            # Add any runtime dependencies here if necessary
          ];
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.default ];
          packages = with pkgs; [
            rustc
            cargo
            rustfmt
            clippy
            # Add any other development tools here
            openssl
            pkg-config
          ];
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
        };
      });
}
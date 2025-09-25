{
  description = "Nix flake for the llm Rust project.";

  inputs = {
    nixpkgs.url = "github:meta-introspector/nixpkgs?ref=feature/CRQ-016-nixify";
    naersk.url = "github:meta-introspector/naersk?ref=feature/CRQ-016-nixify";
    ai-ml-zk-ops.url = "github:meta-introspector/ai-ml-zk-ops?ref=feature/concept-to-nix-8s";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, naersk, ai-ml-zk-ops, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        naersk-lib = naersk.lib.${system};
      in
      {
        packages.llm = naersk-lib.buildPackage {
          pname = "llm";
          version = "1.3.5"; # Get this from Cargo.toml
          src = ./.;
          nativeBuildInputs = with pkgs; [
            pkg-config
            openssl
            # Add any other build dependencies here
          ];
          buildInputs = with pkgs; [
            # Add any runtime dependencies here
          ];
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.llm ];
          packages = with pkgs; [
            rustc
            cargo
            rust-analyzer
            # Add any other development tools here
          ];
        };
      });
}
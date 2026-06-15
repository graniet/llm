{
  description = "A Nix-flake for the log_analyzer Rust project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    naersk.url = "github:nix-community/naersk?ref=master"; # Using nix-community's naersk for now, as meta-introspector's is not directly available.
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, naersk, ... }:
    let
      system = "x86_64-linux"; # Explicitly define the system
      pkgs = nixpkgs.legacyPackages.${system};
      lib = nixpkgs.lib;
      naersk-lib = naersk.lib.${system};
    in
    {
              packages.log-analyzer = naersk-lib.buildPackage {
                pname = "log-analyzer";
                version = "0.1.0";
                src = lib.cleanSource ./.;        nativeBuildInputs = with pkgs; [
          pkg-config
          openssl
        ];
        buildInputs = with pkgs; [
          # Add any runtime dependencies here if necessary
        ];
      };

      devShells.default = pkgs.mkShell {
        inputsFrom = [ self.packages.log-analyzer ];
        packages = with pkgs; [
          rustc
          cargo
          rustfmt
          clippy
          # Add any other development tools here
        ];
        RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
      };
    };
}
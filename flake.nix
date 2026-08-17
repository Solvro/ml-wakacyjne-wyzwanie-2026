{
  description = "Python Environment for Solvro's Machine Learning Summer Course";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Configure Python interpreter + Jupyter kernel support for Molten
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy
          pandas
          matplotlib
          seaborn
          ipykernel     # Required for Molten/Jupyter integration
          jupyter-core
          scikit-learn
          xgboost
        ]);

        # Common C libraries required by native ML Python extensions
        libPath = with pkgs; lib.makeLibraryPath [
          stdenv.cc.cc
          zlib
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.basedpyright
            pythonEnv
          ];

          shellHook = ''
            # Fix library resolution for C/C++ compiled Python packages
            export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"

            # Auto-register this Nix environment as a Jupyter kernel for Molten
            python -m ipykernel install --user --name="solvro-ml" --display-name="Python 3 (Solvro Nix)" > /dev/null 2>&1

            echo "🐍 Solvro ML Nix environment loaded!"
            echo "   • Pandas: $(python -c 'import pandas; print(pandas.__version__)')"
            echo "   • NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
            echo "   • MatPlotLib: $(python -c 'import matplotlib; print(matplotlib.__version__)')"
            echo "   • Seaborn: $(python -c 'import seaborn; print(seaborn.__version__)')"
            echo "   • Scikit-learn: $(python -c 'import sklearn; print(sklearn.__version__)')"
            echo "   • XGBoost: $(python -c 'import xgboost; print(xgboost.__version__)')"
            echo "   • Jupyter Kernel: Registered as 'solvro-ml'"
          '';
        };
      });
}

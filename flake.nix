{
  description = "Python Environment for Solvro's Machine Learning Summer Course";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/0e251e24a4f24e036a084b6b4b2d2491af4167f4";
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
          optuna
          plotly
          nbformat
          pytorch3d
          torchvision
          python-dotenv
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
            echo "$(python -c 'import pandas, numpy, matplotlib, seaborn, sklearn, xgboost, optuna, plotly, nbformat;
            print(f"  • Pandas: {pandas.__version__}")
            print(f"  • NumPy: {numpy.__version__}")
            print(f"  • MatPlotLib: {matplotlib.__version__}")
            print(f"  • Seaborn: {seaborn.__version__}")
            print(f"  • Scikit-learn: {sklearn.__version__}")
            print(f"  • XGBoost: {xgboost.__version__}")
            print(f"  • Optuna: {optuna.__version__}")
            print(f"  • Plotly: {plotly.__version__}")
            print(f"  • NbFormat: {nbformat.__version__}")
            ')"
            echo "  • Jupyter Kernel: Registered as 'solvro-ml'"
          '';
        };
      });
}

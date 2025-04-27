{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSUserEnv {
  name = "julia-fhs";
  targetPkgs = pkgs:
    with pkgs; [
      julia
      gnumake
      gcc
      gfortran
      libatomic_ops
      python3
      perl
      wget
      curl
      m4
      gawk
      patch
      cmake
      pkg-config
      which
    ];

  runScript = "zsh";
}).env

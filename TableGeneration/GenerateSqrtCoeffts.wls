#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]
Needs["ErrorOutput`"]

printerror[computecoeffts[Sqrt, {1, 2}, 6, 25, 15, 11, "sqrt_1_2.txt"]];

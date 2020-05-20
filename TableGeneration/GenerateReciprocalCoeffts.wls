#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]
Needs["ErrorOutput`"]

printerror[computecoeffts[(1/#)&, {1, 2}, 7, 26, 16, 10, "reciprocal_1_2.txt"]];

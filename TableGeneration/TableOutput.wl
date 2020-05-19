(* ::Package:: *)

BeginPackage["TableOutput`"]

writecoeffts::usage = "writecoeffts[channel, c0, c1, c2, t, p, q] writes the coefficients c0, c1, c2 with word lengths t, p, q to the specified output channel. The sign bit is not counted in the word length."

Begin["Private`"]

digitstring::usage = "digitstring[n, d] converts the number n to a binary digit string containing exactly d digits."

binarydigits[n_, d_] :=
	RealDigits[n, 2, d, Max[0, Ceiling[Log[2, Floor[Abs[n]]]]]]

digitstring[n_, d_] :=
	Module[{digits},
		digits = binarydigits[n, d];
		StringJoin[
			If[n < 0, "-", ""],
			Insert[Map[ToString, digits[[1]]], ".", digits[[2]] + 1]
		]
	]

writecoeffts[channel_, c0_, c1_, c2_, t_, p_, q_] :=
	WriteString[channel, digitstring[c0, t], ",", digitstring[c1, p], ",", digitstring[c2, q], "\n"];

End[]

EndPackage[]

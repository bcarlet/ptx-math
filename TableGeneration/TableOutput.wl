(* ::Package:: *)

BeginPackage["TableOutput`"]

writecoeffts::usage = "writecoeffts[channel, c0, c1, c2, t, p, q] writes the coefficients c0, c1, c2 with word lengths t, p, q to the specified output channel. Sign and integer bits are not counted in the word length."

Begin["Private`"]

digitstring::usage = "digitstring[n, d] converts the number n to a binary digit string with exactly d digits after the binary point."

digitstring[n_, d_] :=
	StringJoin[
		If[n < 0, "-", ""],
		Map[ToString, IntegerDigits[IntegerPart[n], 2]],
		".",
		Map[ToString, RealDigits[FractionalPart[n], 2, d, -1][[1]]]
	]

writecoeffts[channel_, c0_, c1_, c2_, t_, p_, q_] :=
	WriteString[channel, digitstring[c0, t], ",", digitstring[c1, p], ",", digitstring[c2, q], "\n"];

End[]

EndPackage[]

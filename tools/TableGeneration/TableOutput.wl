(* ::Package:: *)

BeginPackage["TableOutput`"]

writecoeffts::usage = "writecoeffts[channel, c0, c1, c2, t, p, q] writes the coefficients c0, c1, c2 with word lengths t, p, q to the specified output channel. Sign and integer bits are not counted in the word length."

Begin["Private`"]

binstring::usage = "binstring[n, d] converts the number n to a binary digit string with exactly d digits after the binary point."
hexstring::usage = "hexstring[n, d] converts the number n to a hexadecimal digit string with the precision of d fractional binary digits."
fixedpointstring::usage = "fixedpointstring[n, d] converts the number n to a fixed-point hexadecimal digit string with the precision of d fractional binary digits."

binstring[n_, d_] :=
	StringJoin[
		If[n < 0, "-", ""],
		Map[ToString, IntegerDigits[IntegerPart[n], 2]],
		".",
		Map[ToString, RealDigits[FractionalPart[n], 2, d, -1][[1]]]
	]

hexstring[n_, d_] :=
	StringJoin[
		If[n < 0, "-", ""],
		IntegerString[IntegerPart[n], 16],
		".",
		Map[ToString,
			RealDigits[Floor[Abs[FractionalPart[n]], 2^(-d)], 16, Ceiling[d/4], -1][[1]] /.
				Thread[Range[10, 15] -> CharacterRange["a", "f"]]
		]
	]

fixedpointstring[n_, d_] :=
	If[n < 0, "-", ""] <> IntegerString[IntegerPart[n*2^d], 16]

writecoeffts[channel_, c0_, c1_, c2_, t_, p_, q_] :=
	WriteString[channel, binstring[c0, t], ",", binstring[c1, p], ",", binstring[c2, q], "\n"];

End[]

EndPackage[]

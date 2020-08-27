(* ::Package:: *)

BeginPackage["ErrorOutput`"]

printerror::usage = "printerror[errmax] prints error information as a plain string."

Begin["Private`"]

printerror[errmax_] :=
	Module[{errmaxfmt},
		errmaxfmt = MantissaExponent[errmax, 2];
		Print[
			"Max error: ",
			ToString[errmaxfmt[[1]], InputForm],
			" * 2^",
			ToString[errmaxfmt[[2]], InputForm],
			"\nGood bits: ",
			ToString[Abs[Log[2, errmax]], InputForm]
		];
	]

End[]

EndPackage[]

(* ::Package:: *)

BeginPackage["EnhancedMinimax`"]
Needs["FunctionApproximations`"]
Needs["TableOutput`"]

computecoeffts::usage = "computecoeffts[m, t, p, q] computes the coefficients with word lengths t, p, and q of the quadratic polynomial for square root on (1, 2), to be stored in a look-up table of 2^m entries."

Begin["Private`"]

minimax::usage = "minimax[expr, args] returns the approximation computed by MiniMaxApproximation[expr, args]."
infnorm::usage = "infnorm[expr, x, {a, b}] computes the L-infinity norm of the expression expr in the variable x on the interval [a, b]."
roundbits::usage = "roundbits[x, b] rounds x to b bits after the binary point."

minimax[expr_, args_] :=
	MiniMaxApproximation[expr, args][[2, 1]]

infnorm[expr_, x_, {a_, b_}] :=
	NMaxValue[Abs[expr], {x} \[Element] Interval[{a, b}]]

roundbits[x_, b_] :=
	Round[x, 2^(-b)]

computecoeffts[m_, t_, p_, q_, filename_] :=
	Module[{errmax, expr, i, x, interval, poll, C1, a1, a2, aa2, C2, p0, C0, err, out},
		errmax = 0;
		expr := Sqrt[1+1/(2^m)*i+x];
		interval = {0, 1/(2^m)};

		out = OpenWrite[filename];
		
		For[i = 0, i < 2^m, i++,
			poll = minimax[expr, {x, interval, 2, 0}];
			a1 = Coefficient[poll, x];
			a2 = Coefficient[poll, x, 2];
			C1 = roundbits[a1, p];
			aa2 = a2+(a1-C1)*2^m;
			C2 = roundbits[aa2, q];
			p0 = minimax[expr-C1*x-C2*x^2, {x, interval, 2, 0}]; (* Pineiro uses degree 0, Oberman uses degree 2 *)
			C0 = roundbits[Coefficient[p0, x, 0], t];
			err = infnorm[expr-C0-C1*x-C2*x^2, x, interval];
			errmax = Max[errmax, err];

			writecoeffts[out, C0, C1, C2, t, p, q];
		];

		Close[out];
		
		errmax
	]

End[]

EndPackage[]

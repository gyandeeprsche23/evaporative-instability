(* ::Package:: *)
(* =====================================================================
   Two-layer leaky dielectric under a DC field:
   linear stability, normal-stress balance and neutral curve D0^2(k).

   Continues Leaky_dielectric_DC.nb with the same symbols
     fluid 1: -h1 < z < 0 (wall at potential 0),
     fluid 2:   0 < z < h2 (wall at potential D0),
     interface z = zeta Exp[I k x + s t].
   Run the whole file (Evaluation > Evaluate Notebook, or  << file).

   What changes with respect to the notebook
   -----------------------------------------
   * Base state: the DC interfacial charge is not a free symbol.  d/dt = 0 in
     (2.40) gives  sig1 psi1' = sig2 psi2', which fixes q0 (Part 1 below).
   * Part 3 of the notebook closed the perturbed potentials with
        0 == eps0 eps1 psih1'(0) - eps0 eps2 psih2'(0),
     i.e. zero perturbed charge.  That is the perfect-dielectric condition.
     For a leaky dielectric Gauss' law only DEFINES the perturbed charge qh;
     the condition that closes the potential problem is charge conservation
     (2.48).  At the neutral state (s = 0) it reads
        0 == sig2 psih2'(0) - sig1 psih1'(0) + q0 W'(0)
     (this is your eq248 -- it is a boundary condition for psih, not an
     equation for q, which is why Solve[eq248, q] did not return).
   * The potentials therefore depend on W'(0), and W depends on the potentials
     through the tangential stress.  Both problems are linear, so the loop is
     closed with a 2x2 solve (Part 5).
   ===================================================================== *)
ClearAll["Global`*"];

(* ---------------- Part 1. Base state ---------------- *)
psi1 = a1 z + b1;  psi2 = a2 z + b2;
baseBCs = {
   (psi1 /. z -> -h1) == 0,
   (psi2 /. z ->  h2) == D0,
   (psi1 /. z -> 0) == (psi2 /. z -> 0),
   sig1 D[psi1, z] == sig2 D[psi2, z]};          (* steady current continuity *)
baseCoef = Solve[baseBCs, {a1, b1, a2, b2}][[1]];
psi1base = Simplify[psi1 /. baseCoef];
psi2base = Simplify[psi2 /. baseCoef];
dzpsi1b = Simplify[D[psi1base, z]];
dzpsi2b = Simplify[D[psi2base, z]];
deltaE  = Simplify[dzpsi2b - dzpsi1b];
q0      = Simplify[eps0 (eps1 dzpsi1b - eps2 dzpsi2b)];   (* base surface charge *)
Print["dz psi1^0 = ", dzpsi1b, "   dz psi2^0 = ", dzpsi2b];
Print["q0 = ", q0];

(* ---------------- Part 2. Perturbed potentials ---------------- *)
(* dW0 stands for W'(0); it is eliminated in Part 5. *)
psih1[z_] := C1 Exp[k z] + C2 Exp[-k z];
psih2[z_] := C3 Exp[k z] + C4 Exp[-k z];
potBCs = {
   psih1[-h1] == 0,
   psih2[ h2] == 0,
   psih1[0] - psih2[0] == zeta deltaE,
   sig2 (D[psih2[z], z] /. z -> 0) - sig1 (D[psih1[z], z] /. z -> 0) + q0 dW0 == 0};
potSol = Solve[potBCs, {C1, C2, C3, C4}][[1]];
psih1v[z_] = psih1[z] /. potSol;
psih2v[z_] = psih2[z] /. potSol;

(* ---------------- Part 3. Perturbed velocities (s = 0) ---------------- *)
(* tau stands for the tangential electric stress; eliminated in Part 5.   *)
W1[z_] := aa1 Exp[k z] + bb1 Exp[-k z] + cc1 z Exp[k z] + dd1 z Exp[-k z];
W2[z_] := aa2 Exp[k z] + bb2 Exp[-k z] + cc2 z Exp[k z] + dd2 z Exp[-k z];
velBCs = {
   W1[-h1] == 0, (D[W1[z], z] /. z -> -h1) == 0,
   W2[ h2] == 0, (D[W2[z], z] /. z ->  h2) == 0,
   W1[0] == W2[0],
   (D[W1[z], z] /. z -> 0) == (D[W2[z], z] /. z -> 0),
   mu2 ((D[W2[z], {z, 2}] /. z -> 0) + k^2 W2[0]) -
     mu1 ((D[W1[z], {z, 2}] /. z -> 0) + k^2 W1[0]) == tau,
   W1[0] == 0};                                   (* kinematic condition, s = 0 *)
velSol = Solve[velBCs, {aa1, bb1, cc1, dd1, aa2, bb2, cc2, dd2}][[1]];
W1v[z_] = W1[z] /. velSol;
W2v[z_] = W2[z] /. velSol;

(* ---------------- Part 4. Stresses at the interface ---------------- *)
(* pressure from the x-momentum equation at s = 0:  p = mu (W''' - k^2 W')/k^2 *)
pres[W_, mu_] := mu (D[W, {z, 3}] - k^2 D[W, z])/k^2;
viscN = ((-pres[W2v[z], mu2] + 2 mu2 D[W2v[z], z]) -
         (-pres[W1v[z], mu1] + 2 mu1 D[W1v[z], z])) /. z -> 0;
maxwN = eps0 (eps2 dzpsi2b (D[psih2v[z], z] /. z -> 0) -
              eps1 dzpsi1b (D[psih1v[z], z] /. z -> 0));
NS = viscN + maxwN + (rho2 - rho1) g zeta - gam k^2 zeta;   (* == 0 *)

(* ---------------- Part 5. Close the charge-convection loop ---------------- *)
closure = Solve[{
     dW0 == (D[W1v[z], z] /. z -> 0),
     tau == k^2 q0 (zeta dzpsi1b + psih1v[0])}, {dW0, tau}][[1]];
NSc = (NS /. closure)/zeta;      (* normal stress per unit zeta; function of D0^2 only *)

(* ---------------- Part 6. Neutral curve D0^2(k) ---------------- *)
(* NSc == 0 is a quadratic in Dsq = D0^2.  Parameters are substituted before
   solving (the fully symbolic root is very long).  Exact parameters and a
   100-digit k avoid cancellation between the exponentials.                 *)
neutralDsqAll[kv_?NumericQ, par_] := Module[{expr, num, roots, scale},
   expr = NSc /. D0 -> Sqrt[Dsq] /. Rationalize[par, 0] /. k -> SetPrecision[kv, 100];
   scale = Abs[expr /. Dsq -> 0];                 (* = |(rho2-rho1) g - gam k^2| *)
   num = Numerator[Together[expr]];
   roots = Dsq /. NSolve[num == 0, Dsq, WorkingPrecision -> 30];
   (* keep real positive roots that really satisfy NSc == 0 (the last test
      discards a zero of the denominator if Together did not cancel it)     *)
   Sort[Re /@ Select[roots,
      Abs[Im[#]] <= 10^-12 Abs[#] && Re[#] > 0 && Abs[expr /. Dsq -> #] < 10^-8 scale &]]];
neutralDsq[kv_?NumericQ, par_] := With[{r = neutralDsqAll[kv, par]},
   If[r === {}, Missing["NoNeutralVoltage"], N[First[r]]]];

(* ---------------- Part 7. Fluid pairs (SI units) ---------------- *)
(* bottom fluid (1) / top fluid (2).  Representative literature values; oil
   conductivities vary a lot between batches -- put in your measured ones.  *)
common = {eps0 -> 8.854187817*^-12, g -> 9.81, h1 -> 1.*^-3, h2 -> 1.*^-3};
fluids = <|
   "water / air" -> {eps1 -> 80., eps2 -> 1., sig1 -> 5.5*^-6, sig2 -> 1.*^-15,
     mu1 -> 1.0*^-3, mu2 -> 1.8*^-5, rho1 -> 998., rho2 -> 1.2, gam -> 0.072},
   "castor oil / air" -> {eps1 -> 4.5, eps2 -> 1., sig1 -> 4.*^-11, sig2 -> 1.*^-15,
     mu1 -> 0.69, mu2 -> 1.8*^-5, rho1 -> 961., rho2 -> 1.2, gam -> 0.035},
   "silicone oil / air" -> {eps1 -> 2.8, eps2 -> 1., sig1 -> 1.*^-12, sig2 -> 1.*^-15,
     mu1 -> 0.97, mu2 -> 1.8*^-5, rho1 -> 970., rho2 -> 1.2, gam -> 0.021},
   "water / silicone oil (5 cSt)" -> {eps1 -> 80., eps2 -> 2.7, sig1 -> 5.5*^-6, sig2 -> 1.*^-12,
     mu1 -> 1.0*^-3, mu2 -> 4.6*^-3, rho1 -> 998., rho2 -> 913., gam -> 0.040},
   "silicone oil / castor oil" -> {eps1 -> 2.8, eps2 -> 4.5, sig1 -> 1.*^-12, sig2 -> 4.*^-11,
     mu1 -> 0.97, mu2 -> 0.69, rho1 -> 970., rho2 -> 961., gam -> 0.0045}|>;

(* check: conductor under a perfect dielectric,
   D0^2 = (gam k^2 + (rho1-rho2) g) h2^2 Tanh[k h2]/(eps0 eps2 k)          *)
With[{par = Join[common, fluids["water / air"]], kv = 300.},
  Print["check water/air, k = 300:  ", neutralDsq[kv, par], "   classical: ",
   ((gam k^2 + (rho1 - rho2) g) h2^2 Tanh[k h2]/(eps0 eps2 k)) /. par /. k -> kv]];
(* both should print 1.78309*10^6 V^2 *)

kList = 10.^Range[Log10[20.], Log10[2.*^4], 0.05];
curves = Association[KeyValueMap[
    Function[{name, f}, name -> Select[
        Table[{kv, neutralDsq[kv, Join[common, f]]}, {kv, kList}], NumericQ[#[[2]]] &]], fluids]];

fig = ListLogLogPlot[Values[curves], Joined -> True, PlotLegends -> Keys[curves],
   Frame -> True, FrameLabel -> {"k  (1/m)", "neutral D0^2  (V^2)"},
   PlotLabel -> "DC neutral curves, h1 = h2 = 1 mm (unstable above each curve)",
   PlotRange -> All, GridLines -> Automatic, ImageSize -> 600]

(* effect of the conductivity ratio for one pair; sig1/sig2 = eps1/eps2 gives
   q0 = 0 and recovers the perfect-dielectric result                         *)
ratioCurves = Table[
   With[{par = Join[common, fluids["silicone oil / castor oil"] /. (sig1 -> _) -> (sig1 -> S 4.*^-11)]},
    Select[Table[{kv, neutralDsq[kv, par]}, {kv, kList}], NumericQ[#[[2]]] &]],
   {S, {0.001, 0.025, 0.2, 2.8/4.5, 3., 1000.}}];
figRatio = ListLogLogPlot[ratioCurves, Joined -> True,
   PlotLegends -> ("sig1/sig2 = " <> ToString[#] & /@ {0.001, 0.025, 0.2, "eps1/eps2", 3., 1000.}),
   Frame -> True, FrameLabel -> {"k  (1/m)", "neutral D0^2  (V^2)"},
   PlotLabel -> "silicone oil / castor oil: conductivity ratio", PlotRange -> All, ImageSize -> 600]

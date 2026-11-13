(* ::Package:: *)
(* ======================================================================
   Two-layer leaky dielectric under a DC field: linear stability and neutral curve D0^2 vs k
   ====================================================================== *)

(* Geometry and notation (same symbols as Leaky_dielectric_DC.nb)
      fluid 1 :  -h1 < z < 0     bottom wall z = -h1 held at potential 0
      fluid 2 :    0 < z < h2    top wall z = +h2 held at potential D0
      interface:  z = zeta Exp[I k x + s t],   gravity g along -z,   interfacial tension gam
      eps_j, sig_j, mu_j, rho_j = permittivity (relative), conductivity, viscosity, density of fluid j
   
   Plan of the calculation
      Step 1  base state: potentials, field gradients, interfacial charge q0
      Step 2  perturbed potentials psih1, psih2   (they contain the unknown W'(0))
      Step 3  perturbed velocities W1, W2         (they contain the unknown tangential stress tau)
      Step 4  eliminate W'(0) and tau              (charge convection <-> tangential stress loop)
      Step 5  normal-stress balance at the interface
      Step 6  solve the normal-stress balance for D0^2 at given k  -> neutral curve
      Step 7  fluid properties, check against a classical limit
      Step 8  plots of D0^2 vs k for several fluid pairs, and for varying conductivity ratio
   
   Only the neutral state is computed: growth rate s = 0. All perturbation amplitudes are proportional to zeta. *)

ClearAll["Global`*"];

(* ======================================================================
   Step 1.  Base state (flat interface, steady DC field)
   ====================================================================== *)

(* In each layer the potential obeys Laplace's equation, so it is linear in z:  psi_j = a_j z + b_j.
   The four constants follow from four conditions:
      (i)   psi1(-h1) = 0                      bottom electrode
      (ii)  psi2(+h2) = D0                     top electrode
      (iii) psi1(0) = psi2(0)                  potential is continuous at the interface
      (iv)  sig1 psi1' = sig2 psi2' at z = 0   steady state of the charge equation (2.40):
                                               dq/dt = sig2 psi2' - sig1 psi1' = 0
   Gauss' law then GIVES the base interfacial charge:  q0 = eps0 (eps1 psi1' - eps2 psi2').
   (In the original notebook q was kept as a free symbol; for DC it is fixed by (iv).) *)

psi1 = a1 z + b1;  psi2 = a2 z + b2;
baseBCs = {
   (psi1 /. z -> -h1) == 0,
   (psi2 /. z ->  h2) == D0,
   (psi1 /. z -> 0) == (psi2 /. z -> 0),
   sig1 D[psi1, z] == sig2 D[psi2, z]};
baseCoef = Solve[baseBCs, {a1, b1, a2, b2}][[1]];
psi1base = Simplify[psi1 /. baseCoef];
psi2base = Simplify[psi2 /. baseCoef];
Print["psi1^0(z) = ", psi1base];
Print["psi2^0(z) = ", psi2base];

(* Base field gradients (constant in each layer), their jump deltaE (it forces the perturbed potential in Step 2) and the base interfacial charge q0. Note q0 = 0 when eps1/eps2 = sig1/sig2. *)

dzpsi1b = Simplify[D[psi1base, z]];
dzpsi2b = Simplify[D[psi2base, z]];
deltaE  = Simplify[dzpsi2b - dzpsi1b];
q0      = Simplify[eps0 (eps1 dzpsi1b - eps2 dzpsi2b)];
Print["dz psi1^0 = ", dzpsi1b];
Print["dz psi2^0 = ", dzpsi2b];
Print["deltaE = dz psi2^0 - dz psi1^0 = ", deltaE];
Print["q0 = ", q0];

(* ======================================================================
   Step 2.  Perturbed potentials
   ====================================================================== *)

(* Perturbation psi_j = psi_j^0(z) + psih_j(z) Exp[I k x + s t].  Laplace's equation gives
      psih_j'' - k^2 psih_j = 0   ->   psih_j = C Exp[k z] + C' Exp[-k z].
   Four conditions, linearised about z = 0:
      (i)   psih1(-h1) = 0
      (ii)  psih2(+h2) = 0
      (iii) psih1(0) - psih2(0) = zeta deltaE        continuity of potential on the deformed interface (2.46)
      (iv)  sig2 psih2'(0) - sig1 psih1'(0) + q0 W'(0) = 0
   Condition (iv) is the perturbed charge conservation equation (2.48),
      s qh = sig2 psih2'(0) - sig1 psih1'(0) + q0 W'(0),
   at the neutral state s = 0. The term q0 W'(0) is convection of the base charge by the interfacial flow
   (surface divergence of the velocity = -W'(0)).
   
   IMPORTANT difference from the original notebook: there condition (iv) was
      0 == eps0 eps1 psih1'(0) - eps0 eps2 psih2'(0),
   i.e. zero perturbed charge. That is the perfect-dielectric condition. For a leaky dielectric Gauss' law
   only defines the perturbed charge qh afterwards; the condition that closes the problem is (iv) above
   (your eq248). It is a boundary condition for psih, not an equation for q.
   
   W'(0) is not known yet, so it is carried as the symbol dW0 and eliminated in Step 4. *)

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

(* Check: the four boundary conditions are satisfied (should print {0, 0, 0, 0}). *)

Simplify[{psih1v[-h1], psih2v[h2], psih1v[0] - psih2v[0] - zeta deltaE,
  sig2 (D[psih2v[z], z] /. z -> 0) - sig1 (D[psih1v[z], z] /. z -> 0) + q0 dW0}]

(* ======================================================================
   Step 3.  Perturbed vertical velocities
   ====================================================================== *)

(* Vertical velocity w_j = W_j(z) Exp[I k x + s t]. Eliminating pressure from the momentum equations,
   at s = 0 (inertia drops out) each layer obeys
      (d^2/dz^2 - k^2)^2 W_j = 0   ->   W_j = (aa + cc z) Exp[k z] + (bb + dd z) Exp[-k z].
   Eight conditions:
      (1,2) W1 = W1' = 0 at z = -h1               no slip, no penetration at the bottom wall
      (3,4) W2 = W2' = 0 at z = +h2               same at the top wall
      (5)   W1(0) = W2(0)                         continuity of normal velocity
      (6)   W1'(0) = W2'(0)                       continuity of tangential velocity (u = I W'/k)
      (7)   mu2 (W2'' + k^2 W2) - mu1 (W1'' + k^2 W1) = tau     tangential stress balance at z = 0
      (8)   W1(0) = 0                             kinematic condition s zeta = W(0) with s = 0
   The electric tangential stress is tau = k^2 q0 (zeta dz psi1^0 + psih1(0)): base charge q0 times the
   perturbed tangential field. It is carried as the symbol tau and eliminated in Step 4. *)

W1[z_] := aa1 Exp[k z] + bb1 Exp[-k z] + cc1 z Exp[k z] + dd1 z Exp[-k z];
W2[z_] := aa2 Exp[k z] + bb2 Exp[-k z] + cc2 z Exp[k z] + dd2 z Exp[-k z];
velBCs = {
   W1[-h1] == 0, (D[W1[z], z] /. z -> -h1) == 0,
   W2[ h2] == 0, (D[W2[z], z] /. z ->  h2) == 0,
   W1[0] == W2[0],
   (D[W1[z], z] /. z -> 0) == (D[W2[z], z] /. z -> 0),
   mu2 ((D[W2[z], {z, 2}] /. z -> 0) + k^2 W2[0]) -
     mu1 ((D[W1[z], {z, 2}] /. z -> 0) + k^2 W1[0]) == tau,
   W1[0] == 0};
velSol = Solve[velBCs, {aa1, bb1, cc1, dd1, aa2, bb2, cc2, dd2}][[1]];
W1v[z_] = W1[z] /. velSol;
W2v[z_] = W2[z] /. velSol;

(* ======================================================================
   Step 4.  Close the loop: eliminate W'(0) and tau
   ====================================================================== *)

(* Step 2 gave the potentials in terms of dW0 = W'(0); Step 3 gave the velocities in terms of tau.
   The two definitions
      dW0 = W1'(0)                                   (from the velocity solution, linear in tau)
      tau = k^2 q0 (zeta dz psi1^0 + psih1(0))       (from the potential solution, linear in dW0)
   are two linear equations for the two unknowns. Physically: the tangential electric stress drives an
   interfacial flow, the flow convects the base charge q0, and that changes the perturbed field and hence
   the stress. After this step everything is proportional to zeta and depends on the voltage only via D0^2. *)

closure = Solve[{
     dW0 == (D[W1v[z], z] /. z -> 0),
     tau == k^2 q0 (zeta dzpsi1b + psih1v[0])}, {dW0, tau}][[1]];

(* ======================================================================
   Step 5.  Normal-stress balance at the interface
   ====================================================================== *)

(* Linearised normal-stress jump (fluid 2 minus fluid 1) at z = 0:
      -(p2 - p1) + 2 (mu2 W2' - mu1 W1')        pressure and viscous normal stress
      + eps0 (eps2 dz psi2^0 psih2' - eps1 dz psi1^0 psih1')     Maxwell normal stress  [eps E_z^0 Eh_z]
      + (rho2 - rho1) g zeta                    base hydrostatic pressure evaluated on the deformed interface
      - gam k^2 zeta                            surface tension (curvature)
      = 0
   The perturbed pressure follows from the x-momentum equation at s = 0:
      p_j = mu_j (W_j''' - k^2 W_j') / k^2.
   So the hydrodynamic part of each layer is  -p + 2 mu W' = -mu (W''' - 3 k^2 W')/k^2.
   Signs: gravity is stabilising when the heavier fluid is below (rho1 > rho2), surface tension is always
   stabilising, the Maxwell stress is the destabilising term. *)

pres[W_, mu_] := mu (D[W, {z, 3}] - k^2 D[W, z])/k^2;
viscN = ((-pres[W2v[z], mu2] + 2 mu2 D[W2v[z], z]) -
         (-pres[W1v[z], mu1] + 2 mu1 D[W1v[z], z])) /. z -> 0;
maxwN = eps0 (eps2 dzpsi2b (D[psih2v[z], z] /. z -> 0) -
              eps1 dzpsi1b (D[psih1v[z], z] /. z -> 0));
NS = viscN + maxwN + (rho2 - rho1) g zeta - gam k^2 zeta;

(* Substitute the closure of Step 4 and divide by zeta. NSc == 0 is the neutral-stability condition. It is a rational function of D0^2: (c2 D0^4 + c1 D0^2 + c0)/(1 - beta D0^2), i.e. a quadratic in D0^2. *)

NSc = (NS /. closure)/zeta;

(* ======================================================================
   Step 6.  Solve the normal-stress balance for D0^2 at a given k
   ====================================================================== *)

(* The fully symbolic root is extremely long, so numbers are put in first and the quadratic is then solved.
   neutralDsqAll[k, parameters] does the following:
      1. replaces D0 by Sqrt[Dsq], inserts the parameter values (made exact with Rationalize) and a
         100-digit value of k, so that the large and small exponentials do not cancel badly;
      2. takes the numerator of NSc and solves numerator == 0 for Dsq;
      3. keeps the real, positive roots that really make NSc vanish (this last test removes a zero of the
         denominator in case Together did not cancel it).
   neutralDsq returns the smallest such root = neutral D0^2. The interface is unstable for D0^2 above it. *)

neutralDsqAll[kv_?NumericQ, par_] := Module[{expr, num, roots, scale},
   expr = NSc /. D0 -> Sqrt[Dsq] /. Rationalize[par, 0] /. k -> SetPrecision[kv, 100];
   scale = Abs[expr /. Dsq -> 0];
   num = Numerator[Together[expr]];
   roots = Dsq /. NSolve[num == 0, Dsq, WorkingPrecision -> 30];
   Sort[Re /@ Select[roots,
      Abs[Im[#]] <= 10^-12 Abs[#] && Re[#] > 0 && Abs[expr /. Dsq -> #] < 10^-8 scale &]]];
neutralDsq[kv_?NumericQ, par_] := With[{r = neutralDsqAll[kv, par]},
   If[r === {}, Missing["NoNeutralVoltage"], N[First[r]]]];

(* ======================================================================
   Step 7.  Fluid pairs (SI units) and a check
   ====================================================================== *)

(* Each entry is bottom fluid (1) / top fluid (2), heavier fluid below. Units: sig in S/m, mu in Pa s,
   rho in kg/m^3, gam in N/m, h in m. These are representative literature values; oil conductivities vary
   strongly between samples, so replace them with measured values for your own fluids. Gap thicknesses
   h1, h2 are set in 'common'. *)

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

(* Check against a known result. Water under air is practically a perfect conductor under a perfect
   dielectric, for which
      D0^2 = (gam k^2 + (rho1 - rho2) g) h2^2 Tanh[k h2] / (eps0 eps2 k).
   Both numbers printed below should be 1.78309*10^6 V^2 (k = 300 1/m). *)

With[{par = Join[common, fluids["water / air"]], kv = 300.},
  Print["this analysis : ", neutralDsq[kv, par]];
  Print["classical     : ",
   ((gam k^2 + (rho1 - rho2) g) h2^2 Tanh[k h2]/(eps0 eps2 k)) /. par /. k -> kv]];

(* ======================================================================
   Step 8.  Plots of D0^2 versus k
   ====================================================================== *)

(* Neutral curve for every fluid pair on a logarithmic grid of wavenumbers (20 to 2*10^4 1/m). Above a curve the interface is unstable. For these gaps every curve rises with k, so the long-wave limit k -> 0 gives the critical voltage. *)

kList = 10.^Range[Log10[20.], Log10[2.*^4], 0.05];
curves = Association[KeyValueMap[
    Function[{name, f}, name -> Select[
        Table[{kv, neutralDsq[kv, Join[common, f]]}, {kv, kList}], NumericQ[#[[2]]] &]], fluids]];
fig = ListLogLogPlot[Values[curves], Joined -> True, PlotLegends -> Keys[curves],
   Frame -> True, FrameLabel -> {"k  (1/m)", "neutral D0^2  (V^2)"},
   PlotLabel -> "DC neutral curves, h1 = h2 = 1 mm (unstable above each curve)",
   PlotRange -> All, GridLines -> Automatic, ImageSize -> 600]

(* Critical voltage of each pair, estimated at the smallest k of the grid. *)

TableForm[KeyValueMap[{#1, Sqrt[#2[[1, 2]]]} &, curves],
 TableHeadings -> {None, {"pair (bottom / top)", "D0 critical (V)"}}]

(* Effect of leakiness: silicone oil / castor oil with the conductivity ratio sig1/sig2 varied. When sig1/sig2 = eps1/eps2 there is no interfacial charge (q0 = 0) and the perfect-dielectric result is recovered, which has the highest threshold. *)

ratios = {0.001, 0.025, 0.2, 2.8/4.5, 3., 1000.};
ratioCurves = Table[
   With[{par = Join[common,
       fluids["silicone oil / castor oil"] /. (sig1 -> _) -> (sig1 -> S 4.*^-11)]},
    Select[Table[{kv, neutralDsq[kv, par]}, {kv, kList}], NumericQ[#[[2]]] &]],
   {S, ratios}];
figRatio = ListLogLogPlot[ratioCurves, Joined -> True,
   PlotLegends -> ("sig1/sig2 = " <> ToString[#] & /@ {0.001, 0.025, 0.2, "eps1/eps2", 3., 1000.}),
   Frame -> True, FrameLabel -> {"k  (1/m)", "neutral D0^2  (V^2)"},
   PlotLabel -> "silicone oil / castor oil: effect of conductivity ratio",
   PlotRange -> All, GridLines -> Automatic, ImageSize -> 600]


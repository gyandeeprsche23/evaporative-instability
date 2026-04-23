(* ================================================================
   Marangoni Thin-Film Model with Contact Angle Effects
   Extension of Pillai & Narayanan (2018), Fluid Dyn. Res. 50, 051411
   Key extension: disjoining pressure Pi(h; theta_c) added to normal
   stress balance, parameterizing partial wetting via contact angle theta_c.
   ================================================================ *)

(* ---- 1. Physical Parameters (Water-Vapor, 100C, Table 1 of paper) ---- *)
kl = 6.8*^-1; kv = 2.5*^-2;          (* W/(m.K) *)
gam0 = 5.8*^-2; gamT = 2.0*^-4;      (* N/m, N/(m.K) *)
DeltaT = 10.;                          (* deg C imposed Delta T *)
kappa = kv/kl;                         (* ~0.037 *)
MaNum = gamT*DeltaT/gam0;             (* ~0.034 Marangoni number *)
Print["kappa = ", N[kappa, 4], "   Ma = ", N[MaNum, 4]]

(* Simulation setup (Marangoni case: Hl=5mm, Hv=15mm, Fig.8b of paper) *)
delta  = 0.05;        (* precursor thickness h*/Hl; increased for stability *)
kwave  = 0.08;        (* wavenumber (one-wavelength domain) *)
Ldom   = 2*Pi/kwave;  (* ~78.5 Hl *)
tFin   = 2500;        (* final dimensionless time *)
nPts   = 128;         (* spatial grid points *)

(* ---- 2. Disjoining Pressure and Flux Functions ---- *)
(* Pi(h; theta_c): repulsive for h < delta, attractive for h > delta *)
PiDP[h_, tc_]   := (tc^2/delta)*((delta/h)^3 - (delta/h)^2)
PiDPd[h_, tc_]  := tc^2*(-3*delta^2/h^4 + 2*delta/h^3)    (* d Pi/dh *)
PiDPdd[h_, tc_] := tc^2*(12*delta^2/h^5 - 6*delta/h^4)    (* d^2 Pi/dh^2 *)

(* G(h; theta_c) = h^3/3 * Pi'(h) + Ma * h^2/2 and dG/dh *)
Gf[h_, tc_]  := h^3/3*PiDPd[h, tc] + MaNum*h^2/2
Gfd[h_, tc_] := h^2*PiDPd[h, tc] + h^3/3*PiDPdd[h, tc] + MaNum*h

(* ---- 3. Interface Temperature and Heat Flux ---- *)
(* Theta_int from steady bilayer conduction: kl(1-Theta)/h = kv*Theta/(1-h) *)
ThetaI[h_] := (1 - h)/(1 - h + kappa*h)

(* J(h) = kappa/(1-(1-kappa)h): higher J where liquid is thick (thin vapor) *)
Jflux[h_] := kappa/(1 - (1 - kappa)*h)

(* ---- 4. Linear Stability Analysis ---- *)
(* Linearize h = 1 + eps*e^{sigma t + ikx}:
   sigma(k) = G(1; theta_c)*k^2 - k^4/3
   Unstable when k < k_c = Sqrt(3*G(1; theta_c)) -- long waves destabilize *)
sigmaLin[k_, tc_] := Gf[1, tc]*k^2 - k^4/3

Print["\n---- Linear Growth Rate sigma at k = ", kwave, " ----"]
Grid[Prepend[
  Table[{ToString[tc] <> "\[Degree]", N[Gf[1, tc*Degree], 4],
    N[sigmaLin[kwave, tc*Degree], 5],
    N[Sqrt[3*Gf[1, tc*Degree]], 4]}, {tc, {0, 10, 20, 30}}],
  Style[#, Bold] & /@ {"\[Theta]c", "G(1;\[Theta]c)",
    "\[Sigma](k=0.08)", "k_c"}], Frame -> All]

figDisp =
 Plot[Evaluate@
   Table[{sigmaLin[k, tc*Degree], 0}, {tc, {0, 10, 20, 30}}], {k, 0,
   0.35}, PlotLegends ->
   Placed[{"\[Theta]c=0\[Degree]", "", "\[Theta]c=10\[Degree]", "",
     "\[Theta]c=20\[Degree]", "", "\[Theta]c=30\[Degree]", ""},
    Right], PlotStyle ->
   Flatten@Table[{Directive[col, Thick],
      Directive[col, Dotted, Thin]}, {col, {Black, Blue, Red,
       Green}}], AxesLabel -> {"k", "\[Sigma](k)"},
  PlotLabel -> "Dispersion Relation for Various Contact Angles",
  Frame -> True, GridLines -> Automatic, PlotRange -> {-0.002, 0.005},
   ImageSize -> 450]

(* ---- 5. Nonlinear PDE Solver ---- *)
(*
  dh/dt = h^3/3 * h_xxxx + h^2 * h_x * h_xxx
          - Gfd(h; theta_c) * h_x^2 - Gf(h; theta_c) * h_xx
  IC:  h(0, x) = 1 + 0.01*Cos(kwave*x)
  BC:  periodic on [0, Ldom]

  FIX 1: PeriodicBoundaryCondition is removed because it forces Mathematica
  to use the FEM spatial discretization, which rejects the TensorProductGrid
  options (MinPoints, MaxPoints, DifferenceOrder).  The Fourier pseudospectral
  method selected via DifferenceOrder -> "Pseudospectral" is inherently
  periodic on [0, Ldom], so no explicit BC is needed.

  FIX 2: MaxSteps -> 1000000 (explicit integer literal) avoids the ioppf
  error that occurs when 10^6 is promoted to a machine-precision float in
  a mixed-precision evaluation context.
*)
SolveFilm[thetaC_?NumericQ, tmax_: tFin, ng_: nPts] :=
 NDSolve[{D[h[t, x], t] ==
    h[t, x]^3/3*D[h[t, x], {x, 4}] +
     h[t, x]^2*D[h[t, x], x]*D[h[t, x], {x, 3}] -
     Gfd[h[t, x], thetaC]*D[h[t, x], x]^2 -
     Gf[h[t, x], thetaC]*D[h[t, x], {x, 2}],
   h[0, x] == 1 + 0.01*Cos[kwave*x]},
  h, {t, 0, tmax}, {x, 0, Ldom},
  Method -> {"MethodOfLines",
    "SpatialDiscretization" -> {"TensorProductGrid",
      "MinPoints" -> ng, "MaxPoints" -> ng,
      "DifferenceOrder" -> "Pseudospectral"}},
  MaxSteps -> 1000000,
  PrecisionGoal -> 4]

(* ---- 6. Run Simulations ---- *)
tcList = {0, 10, 20, 30}*Degree;  (* contact angles in radians *)
tcTags = {"\[Theta]c=0\[Degree]", "\[Theta]c=10\[Degree]",
  "\[Theta]c=20\[Degree]", "\[Theta]c=30\[Degree]"};
pStyles = {Directive[Black, Thick], Directive[Blue, Thick],
   Directive[Red, Thick], Directive[Darker@Green, Thick]};

Print["\n---- Running Simulations ----"]
allSols =
  Table[Print["  Solving \[Theta]c = ", Round[tc*180/Pi, 1], "\[Degree]..."];
   {tc, SolveFilm[tc]}, {tc, tcList}];
Print["Complete."]

(* ---- 7. Post-Processing and Figures ---- *)
xFine = Range[0., Ldom, Ldom/(5 nPts)];  (* fine x-grid for evaluation *)

(* 7a. h(x,t) profiles at two time snapshots *)
FilmProfile[t_] :=
 Plot[Evaluate@Table[h[t, x] /. allSols[[i, 2]], {i, 4}], {x, 0,
   Ldom}, PlotLegends -> Placed[tcTags, Right], PlotStyle -> pStyles,
  AxesLabel -> {"x/Hl", "h(x,t)"}, Frame -> True,
  PlotLabel -> Row[{"Film Profile  t = ", t}], GridLines -> Automatic,
   PlotRange -> {0, 2}, ImageSize -> 460]

(* 7b. Heat flux J(x,t) profiles *)
HeatFluxProfile[t_] :=
 Plot[Evaluate@Table[Jflux[h[t, x] /. allSols[[i, 2]]], {i, 4}], {x,
   0, Ldom}, PlotLegends -> Placed[tcTags, Right],
  PlotStyle -> pStyles,
  AxesLabel -> {"x/Hl",
    "J(x,t)  [kl\[CenterDot]\[CapitalDelta]T/Hl]"},
  PlotLabel -> Row[{"Heat Flux Distribution  t = ", t}],
  Frame -> True, GridLines -> Automatic, ImageSize -> 460]

(* 7c. Time evolution of min film thickness (rupture indicator) *)
tSamp = Union[{0}, Range[200, tFin, 200]];

hMinData =
  Table[Quiet@{tc,
     Table[{t, Min@Table[h[t, x] /. allSols[[i, 2]], {x, xFine}]}, {t,
        tSamp}]}, {i, 4}];

figHmin =
 ListLinePlot[hMinData[[All, 2]],
  PlotLegends -> Placed[tcTags, Right], PlotStyle -> pStyles,
  AxesLabel -> {"t", "hmin(t)"},
  PlotLabel ->
   "Minimum Film Thickness vs Time\n(Contact angle effect on rupture)",
  Frame -> True, GridLines -> Automatic,
  PlotRange -> {0, 1.5}, ImageSize -> 460]

(* 7d. Peak heat flux vs time *)
JmaxData =
  Table[Quiet@
    Table[{t,
      Max@Table[Jflux[h[t, x] /. allSols[[i, 2]]], {x, xFine}]}, {t,
      tSamp}], {i, 4}];

figJmax =
 ListLinePlot[JmaxData, PlotLegends -> Placed[tcTags, Right],
  PlotStyle -> pStyles, AxesLabel -> {"t", "Jmax(t)"},
  PlotLabel -> "Peak Heat Flux Evolution vs Time", Frame -> True,
  GridLines -> Automatic, ImageSize -> 460]

(* 7e. Heat flux at trough vs theta_c (parametric summary) *)
JTroughVsAngle =
  Table[hTrough = Min@Table[h[tFin, x] /. allSols[[i, 2]], {x, xFine}];
   {tcList[[i]]*180/Pi, Jflux[hTrough]}, {i, 4}];

figJvsTheta =
 ListPlot[JTroughVsAngle, PlotStyle -> {PointSize[0.025], Blue},
  Joined -> True,
  AxesLabel -> {"\[Theta]c  (degrees)",
    "J at trough  [kl\[CenterDot]\[CapitalDelta]T/Hl]"},
  PlotLabel ->
   "Heat Flux at Trough Location vs Contact Angle\nat t = " <>
    ToString[tFin], Frame -> True, GridLines -> Automatic,
  ImageSize -> 360]

(* 7f. Space-time density plots of h(x,t) *)
STplots =
  Table[DensityPlot[
    h[t, x] /. allSols[[i, 2]], {x, 0, Ldom}, {t, 0, tFin},
    ColorFunction -> "TemperatureMap", PlotLegends -> Automatic,
    FrameLabel -> {"x/Hl", "t"}, PlotLabel -> tcTags[[i]],
    PlotPoints -> 60, ImageSize -> 210], {i, 4}];

(* ---- 8. Display All Results ---- *)
Print@FilmProfile[tFin/2]
Print@FilmProfile[tFin]
Print@HeatFluxProfile[tFin]
Print@figHmin
Print@figJmax
Print@figJvsTheta
Print@Row[STplots, Spacer[4]]

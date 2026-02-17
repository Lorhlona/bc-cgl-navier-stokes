---
marp: true
theme: default
paginate: true
math: katex
---

# LoNalogy 詳細ログ（Marp数式版）

- この版は **Marp表示用**
- 数式は KaTeX 形式（`math: katex`）

---

## 0. 代数の基礎

$$
i^2=-1,\quad j^2=+1,\quad [i,j]=0
$$

$$
e_\pm=\frac{1\pm j}{2},\quad
e_\pm^2=e_\pm,\quad
e_+e_-=0,\quad
j=e_+-e_-
$$

---

## 0.1 記号の固定（衝突回避）

- $\Xi(\mathbf x,t)$: LCMSのセクター分裂係数（$\sigma_3$前の係数）
- $\mathcal{Y}(s)$: 境界アドミタンス演算子（旧 $\Lambda(s)$）
- $\Omega_\Lambda$: 宇宙定数密度パラメータ

---

## 1. 出発点（波動関数）

$$
\Psi=\psi_+e_+ + \psi_-e_-
$$

$$
\psi_+=\sqrt{p_+}\,e^{iS_+},\quad
\psi_-=\sqrt{p_-}\,e^{iS_-}
$$

---

## 2. 双複素ボルン則

$$
P:=|\Psi|^2=|\psi_+|^2e_+ + |\psi_-|^2e_- = p_+e_+ + p_-e_-
$$

---

## 3. 双複素CGL（初期主方程式）

$$
\partial_t\Psi=(\alpha+i\omega+j\kappa+ij\lambda)\Psi
+B\nabla^2\Psi + C|\Psi|^2\Psi
$$

---

## 4. e+/e- 分解

$$
\partial_t\psi_\pm=
\bigl[(\alpha\pm\kappa)+i(\omega\pm\lambda)\bigr]\psi_\pm
+B_\pm\nabla^2\psi_\pm
+C_\pm|\psi_\pm|^2\psi_\pm
$$

---

## 5. 試行錯誤1（当初の主張）

当初の主張:
- `kappa > 0` で `p- -> p+`
- `kappa < 0` で `p+ -> p-`

当初よく書いた式:

$$
\partial_t p_+ + \nabla\cdot J_+ = +2\kappa p_+,\quad
\partial_t p_- + \nabla\cdot J_- = -2\kappa p_-
$$

---

## 6. 修正A（重要）

修正結論:
- `j kappa Psi` は **対角作用**
- まず出るのは `(+/-)` の成長率差
- **kappa単独ではセクター間移送を作らない**

要点式:

$$
\text{kappa term} \Rightarrow (\alpha+\kappa),(\alpha-\kappa)\ \text{の差}
$$

---

## 7. 修正B（連続方程式）

一般形に修正:

$$
\partial_t p_\pm + \nabla\cdot J_\pm = R_\pm
$$

`R±` には少なくとも:
- `2(alpha±kappa)p±`
- `Re(B±), Re(C±)` 由来の項

---

## 8. 修正C（本当に移送を入れる式）

移送を主張するなら off-diagonal が必要:

$$
\partial_t p_+ + \nabla\cdot J_+ = \Gamma p_- - \Gamma p_+
$$
$$
\partial_t p_- + \nabla\cdot J_- = -\Gamma p_- + \Gamma p_+
$$

---

## 9. 総量保存の確認

$$
\partial_t(p_+ + p_-) + \nabla\cdot(J_+ + J_-) = 0
$$

---

## 10. LCMS vs CGL（レイヤ整理）

- 基礎: **LCMS（閉体系）**
- 有効: **双複素CGL（開放系・粗視化）**

$$
\text{LCMS (closed)}\ \to\ \text{coarse-grain}\ \to\ \text{bicomplex CGL (open)}
$$

---

## 11. 標準量子の回収

回収条件（典型）:
- `Gamma = 0`
- `psi- = 0`
- `B` を通常のSchrodinger係数に同定

すると標準QM（Schrodinger + Born + 確率保存）を回収。

---

## 12. 他分野整合（要点）

- 古典: `hbar -> 0` で WKB/HJ
- 統計/熱: 交換モデルで詳細釣り合い
- 電磁気: 最小結合 `D_mu = partial_mu + i q A_mu / hbar`
- 相対論: Dirac/KG共変作用へ
- 宇宙論: 5/27/68 は確率でなくエネルギー密度比

---

## 13. 世代数3（現状）

交差ブレーンの基本:

$$
N_{\text{gen}} = |I_{ab}|
$$

$$
I_{ab}=\prod_i\left(n_a^i m_b^i - m_a^i n_b^i\right)
$$

現状:  
「3世代必然化」は整数制約問題として定式化済み。  
ただし具体コンパクト化で `g=1,2` 排除の完了証明は未達。

---

## 14. 工学拡張（境界回路）

境界入出力:

$$
I(s)=\mathcal{Y}(s)V(s)+I_0(s)
$$

テブナン演算子版:

$$
Z_{\text{th}}(s)=\mathcal{Y}(s)^{-1},\quad
V_{\text{th}}(s)=\mathcal{Y}(s)^{-1}I_0(s)
$$

---

## 15. LoNalogy二層の境界式

$$
\begin{bmatrix}I_+\\I_-\end{bmatrix}
=
\begin{bmatrix}
\mathcal{Y}_{++} & \mathcal{Y}_{+-}\\
\mathcal{Y}_{-+} & \mathcal{Y}_{--}
\end{bmatrix}
\begin{bmatrix}V_+\\V_-\end{bmatrix}
+
\begin{bmatrix}I_{0+}\\I_{0-}\end{bmatrix}
$$

可視有効演算子:

$$
\mathcal{Y}_{\text{eff}}
=
\mathcal{Y}_{++}
-\mathcal{Y}_{+-}\mathcal{Y}_{--}^{-1}\mathcal{Y}_{-+}
$$

---

## 16. 数値予言フェーズ（現在）

未完タスク:
1. `delta Q` を含む2流体摂動の完全実装  
2. `H(z), f sigma8(z), S8` 同時フィット  
3. 最小実験系で逆問題検証

ここを詰めれば「理論」から「予言」へ進む。

---

## 17. LCMS 基本式（非相対論・明示形）

二重項:

$$
\psi=
\begin{bmatrix}
\psi_+\\
\psi_-
\end{bmatrix},
\quad
\Psi=\psi_+e_+ + \psi_-e_-
$$

閉体系LCMSの最小形:

$$
i\hbar \partial_t \psi
=
\left[
\hat H_0\,\mathbf I
+\hbar \Gamma(\mathbf x,t)\sigma_1
+\hbar \Xi(\mathbf x,t)\sigma_3
\right]\psi
$$

$$
\hat H_0=
-\frac{\hbar^2}{2m}\left(\nabla-\frac{i q}{\hbar}\mathbf A\right)^2+V
$$

観測写像:

$$
P=|\Psi|^2=p_+e_+ + p_-e_-,
\quad p_\pm=|\psi_\pm|^2,
\quad \pi(P)=p_+
$$

---

## 17. LCMS 基本式（数学的要件）

$$
\mathcal H=L^2(\Omega)\otimes\mathbb C^2,\quad
\psi(t)\in D(\hat H)\subset\mathcal H
$$

$$
\hat H=\hat H^\dagger
\ \Longrightarrow\
\psi(t)=e^{-i\hat H t/\hbar}\psi(0)
$$

$$
\|\psi(t)\|_{\mathcal H}^2=\|\psi(0)\|_{\mathcal H}^2
$$

（閉体系LCMSでは自己共役ハミルトニアンを基本仮定に置く）

---

## 18. LCMS 連続方程式（交換を含む）

### 18.1 厳密式（閉体系・ユニタリ）

`17` のハミルトニアン型 LCMS から厳密に出る式:

$$
\partial_t p_+ + \nabla\!\cdot J_+
=
2\,\Gamma\,\mathrm{Im}(\psi_+^\ast\psi_-)
$$

$$
\partial_t p_- + \nabla\!\cdot J_-
=
-2\,\Gamma\,\mathrm{Im}(\psi_+^\ast\psi_-)
$$

したがって

$$
\partial_t(p_+ + p_-) + \nabla\!\cdot(J_+ + J_-) = 0
$$

ここで右辺は `p_+-p_-` ではなく、コヒーレンス
$\mathrm{Im}(\psi_+^\ast\psi_-)$ で決まる。

---

## 18. LCMS 連続方程式（交換を含む）

### 18.2 交換レート方程式（粗視化近似）

次はデコヒーレンス後の有効式:

$$
\partial_t p_+ + \nabla\!\cdot J_+ = \Gamma(p_- - p_+)
$$

$$
\partial_t p_- + \nabla\!\cdot J_- = \Gamma(p_+ - p_-)
$$

注意:
- `18.1` が厳密、`18.2` は近似
- 移送の起源は off-diagonal（$\sigma_1$）
- 対角項（$\sigma_3$）は主に位相差/分裂の源

---

## 19. LCMS 相対論版（明示形）

$$
\left(i\hbar c\,\gamma^\mu D_\mu - mc^2\right)\psi
-\hbar c\,\Gamma(\Theta)\sigma_1\psi
-\hbar c\,(\partial_\mu\Theta)\gamma^\mu\sigma_3\psi
=0
$$

$$
D_\mu=\partial_\mu+\frac{i q}{\hbar}A_\mu
$$

対応する作用（代表形）:

$$
\mathcal L=
\bar\psi(i\hbar c\gamma^\mu D_\mu-mc^2)\psi
-\hbar c\,\Gamma\,\bar\psi\sigma_1\psi
-\hbar c\,(\partial_\mu\Theta)\bar\psi\gamma^\mu\sigma_3\psi
+\frac{f^2}{2}\partial_\mu\Theta\partial^\mu\Theta-U(\Theta)
$$

---

## 20. 古典力学との接続

WKB近似:

$$
\psi_\pm=\sqrt{\rho_\pm}\,e^{iS_\pm/\hbar}
$$

$\hbar\to 0$ 極限で各セクターは

$$
\partial_t S_\pm + H(\mathbf x,\nabla S_\pm,t)=0
$$

へ接続（Hamilton-Jacobi）。

有効質点像:
- 古典軌道は $S_\pm$ の特性曲線
- $\Gamma\neq 0$ で可視/不可視セクター間の占有が再配分

---

## 21. 熱力学・統計との接続

空間を落とした最小交換系:

$$
\dot p_+ = \Gamma(p_- - p_+),\quad
\dot p_- = \Gamma(p_+ - p_-)
$$

エントロピー（2状態）:

$$
S=-(p_+\ln p_+ + p_-\ln p_-)
$$

このとき

$$
\dot S
=
\Gamma (p_- - p_+)\ln\!\frac{p_-}{p_+}\ge 0
$$

---

## 21. 熱力学・統計との接続（補足）

要点:
- 交換は詳細釣り合い型にできる
- 「見え方の変化」を熱力学と矛盾なく実装可能

補足（数学的に厳密な言い方）:
- 2状態マルコフ過程（生成行列は Metzler, 行和0）
- 定常分布 $\pi$ に対して相対エントロピー
  $D_{\mathrm{KL}}(p\Vert\pi)$ は単調減少

---

## 22. 解析量子（標準QM）との接続

標準量子を回収する条件:

1. $\Gamma=0$  
2. 初期条件 $\psi_-(t_0)=0$  
3. 通常の $\hat H_0$（Hermitian）

すると不変部分空間上で

$$
i\hbar \partial_t \psi_+ = \hat H_0\psi_+
$$

$$
p=\pi(P)=|\psi_+|^2,\quad
\partial_t p + \nabla\!\cdot J = 0
$$

---

## 22. 解析量子（測定写像の厳密化）

$$
\rho \mapsto \mathcal E(\rho),\quad
\mathcal E:\text{CPTP map}
$$

可視確率を POVM で書くと

$$
p_{\mathrm{vis}}=\mathrm{Tr}(M_+\rho),\quad
M_+\ge 0,\ \sum_k M_k=\mathbf I
$$

$\pi(P)=p_+$ は、特定POVM選択の有効表現として読む。

---

## 23. 電磁気との接続

最小結合:

$$
\nabla \to \nabla-\frac{i q}{\hbar}\mathbf A,\quad
\partial_t\to \partial_t+\frac{i q}{\hbar}\phi
$$

要件:
- ゲージ不変性を保つ
- 可視電荷の連続式を壊さない
- 破れが見える場合は「有効理論での環境流出」を明示

閉体系LCMSでは

$$
\partial_\mu J^\mu_{\mathrm{tot}}=0
$$

を基本に置くのが安全。

---

## 23. 電磁気との接続（二層電荷）

二層結合の典型:

$$
D_\mu\psi=
\left(\partial_\mu+\frac{i}{\hbar}Q A_\mu\right)\psi,\quad
Q=\begin{bmatrix}q_+&0\\0&q_-\end{bmatrix}
$$

可視のみを電磁結合させる最小モデルは $q_-=0$。

---

## 24. 相対論との接続

方針:
- CGLを基礎にしない（非相対論拡散型）
- 基礎は共変作用（Dirac/KG + 二重項）

整合条件:
- Lorentz共変
- 因果性
- エネルギー運動量テンソルの定義可能性

開放効果は:
- 基礎作用でなく、有効粗視化後の項として導入

---

## 24. 相対論との接続（実務条件）

実務上の要件:
- 有効化する際は GKSL 形式（完全正値）を維持
- 局所性条件を明記し、超光速信号を回避

---

## 25. 宇宙論との接続（FRW）

背景方程式（代表）:

$$
3M_{\mathrm{Pl}}^2H^2=\rho_r+\rho_\Lambda+\rho_+ + \rho_- + \rho_\Theta
$$

交換項:

$$
\dot\rho_+ + 3H(1+w_+)\rho_+ = +Q
$$
$$
\dot\rho_- + 3H(1+w_-)\rho_- = -Q
$$

最小例:

$$
Q=\Gamma(\Theta)(\rho_- - \rho_+)
$$

---

## 25. 宇宙論との接続（共変保存形）

$$
\nabla_\mu T^{\mu\nu}_{(+)}=Q^\nu,\quad
\nabla_\mu T^{\mu\nu}_{(-)}=-Q^\nu
$$

$$
\nabla_\mu\!\left(T^{\mu\nu}_{(+)}+T^{\mu\nu}_{(-)}\right)=0
$$

FRW背景では通常

$$
Q^\nu = Q\,u^\nu
$$

を採用して背景方程式へ落とす。

---

## 25. 宇宙論との接続（注意と摂動）

注意:
- 5/27/68 はエネルギー密度比
- 直接 $p_\pm$ と同一視しない

摂動レベル（ニュートンゲージ）最小拡張:

$$
\dot\delta_i
+(1+w_i)(\theta_i-3\dot\Phi)
+3H(c_{s,i}^2-w_i)\delta_i
=
\frac{\delta Q_i}{\rho_i}-\frac{Q_i}{\rho_i}\delta_i,\quad i=\pm
$$

$H(z)$ と $f\sigma_8(z)$ の同時適合には、
背景 $Q$ と摂動 $\delta Q$ の整合実装が必須。

---

## 26. 素粒子論との接続

可視/暗黒二重項をQFTで表現:

$$
\psi=
\begin{bmatrix}
\psi_+\\
\psi_-
\end{bmatrix}
$$

代表ラグランジアン:

$$
\mathcal L=\mathcal L_{\mathrm{SM}}
+\bar\psi(i\gamma^\mu\partial_\mu-M)\psi
-\Gamma\,\bar\psi\sigma_1\psi
+\mathcal L_{\mathrm{portal}}
$$

---

## 26. 素粒子論との接続（実装論点）

実装上の論点:
- anomaly cancellation（ベクトルライク配置）
- ポータル強度制約
- 暗黒安定性（対称性 or 凍結機構）

最小可 renormalizable ポータル（次元4以下）:

$$
\mathcal L_{\mathrm{portal}}^{\mathrm{baseline}}
=
\lambda_{HS}|H|^2|S|^2
$$

（基礎検証ではまず1種類に固定。  
kinetic-mixing と neutrino-portal は拡張として別扱い。）

---

## 27. 弦理論との接続

交差数が世代数:

$$
N_{\mathrm{gen}}=|I_{ab}|,
\quad
I_{ab}=\prod_i(n_a^i m_b^i - m_a^i n_b^i)
$$

ハイパーチャージ無質量条件（典型）:

$$
\sum_x x_x N_x B_x^I = 0\quad (\forall I)
$$

---

## 27. 弦理論との接続（整合条件）

タドポール条件（典型）:

$$
\sum_a N_a\left([\Pi_a]+[\Pi_{a'}]\right)-4[\Pi_{O6}] = 0
$$

K理論制約（典型）:

$$
\sum_a N_a\,[\Pi_a]\cdot[\Pi_{\mathrm{probe}}]\equiv 0\pmod 2
$$

---

## 27. 弦理論との接続（狙い）

接続の狙い:
- even/odd 分解と $e_\pm$ 対応
- 余剰U(1)の質量化
- 3世代必然化を整数制約問題として詰める

---

## 27. 弦理論との接続（主張スコープ）

ここでの主張は「一般定理」ではなく:

- 特定コンパクト化クラスでの実現可能性
- 追加制約（タドポール, K理論, Yukawa選択則）下の最小解探索

したがって「3世代必然化」は **条件付き命題**として扱う。

---

## 28. 逆問題・工学との接続

境界入出力演算子:

$$
I(s)=\mathcal{Y}(s)V(s)+I_0(s)
$$

二層化:

$$
\begin{bmatrix}I_+\\I_-\end{bmatrix}
=
\begin{bmatrix}
\mathcal{Y}_{++} & \mathcal{Y}_{+-}\\
\mathcal{Y}_{-+} & \mathcal{Y}_{--}
\end{bmatrix}
\begin{bmatrix}V_+\\V_-\end{bmatrix}
+
\begin{bmatrix}I_{0+}\\I_{0-}\end{bmatrix}
$$

---

## 28. 逆問題・工学との接続（可視有効演算子）

可視有効演算子:

$$
\mathcal{Y}_{\mathrm{eff}}
=
\mathcal{Y}_{++}
-\mathcal{Y}_{+-}\mathcal{Y}_{--}^{-1}\mathcal{Y}_{-+}
$$

意味:
- 見えないセクター情報が可視応答へ混入
- 逆散乱で不可視側の地形推定が可能

数学的要件:
- $\mathcal{Y}(s)$ の正実性（受動性）
- 因果性（解析性, Kramers-Kronig）
- 一意性/安定性条件（測定配置依存）

---

## 29. どこまで確立し、何が未完か

確立寄り:
- 二重項 + 射影 + 交換という数理骨格
- LCMS（基礎）/CGL（有効）の分離

---

## 29. どこまで確立し、何が未完か（未完）

未完:
- 交換入り摂動の精密数値（$H(z),f\sigma_8,S_8$同時）
- 3世代必然化の完全証明（具体コンパクト化で $g=1,2$ 排除）
- 実験設計まで落とした逆問題検証

---

## 30. 分野展開の数学テンプレ（実装用）

分野ごとに次の順を固定すると破綻しにくい:

1. **状態空間**（例: $\mathcal H=L^2(\Omega)\otimes\mathbb C^2$）  
2. **閉体系力学**（例: $i\hbar\dot\psi=\hat H\psi$）  
3. **観測写像**（例: $\pi(\rho)=\mathrm{Tr}(P_+\rho)$）  
4. **有効化**（GKSL, coarse-grain, adiabatic elimination）  
5. **検証量**（保存量・応答関数・尤度）

この順序を守ると、分野間で式の互換性を保ちやすい。

---

## 31. 研究計画（数理優先）

### Phase A（厳密化）
- LCMS の well-posedness（存在・一意・安定性）
- `18.2` 近似の導出条件（時間スケール分離）

### Phase B（分野別）
- 宇宙論: $Q,\delta Q$ を同時推定
- QFT: ポータル1種固定で拘束
- 逆問題: $\mathcal{Y}_{\mathrm{eff}}$ の識別可能性解析

### Phase C（実証）
- 同一パラメータで複数観測を同時適合
- 未使用データへの外挿予言で検証

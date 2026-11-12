# Project Video Presentation Script (3–5 Minutes)
**Course:** Advanced Data Analytics (ADA-2026) — Jackfruit Project Review  
**Project:** *Does the Way We Transform Outcomes Change Who We Target? Scale-Dependence of CATE and Uplift Rankings for Zero-Heavy Outcomes, with Count-Aware Learners*  
**Team Members:**
* **Pallavi B** (`PES1UG24AM906`)
* **Khushi Patel** (`PES1UG24AM134`)
* **N Lakshanya** (`PES1UG24AM169`)
* **Porla Neha** (`PES1UG25AM808`)

**Target Duration:** Exactly 4 minutes 15 seconds (Comfortably within the 3–5 minute limit)  
**Slide Deck:** `4_Project_PPT/Rescaled_Comprehensive_Presentation.pptx`

---

## Recording Guidelines for the Team
* **Recommended Software:**
  1. **PowerPoint Built-in Recording:** Open `Rescaled_Comprehensive_Presentation.pptx` in Microsoft PowerPoint → Click **Record** tab → **From Beginning**. Record your audio and camera thumbnail, then click **Export to Video** (.mp4, 1080p).
  2. **Zoom / Google Meet Self-Recording:** Start an empty meeting with your camera and mic on, share the full screen, and record to cloud or computer.
  3. **Loom / OBS Studio:** Free screen + webcam recording.
* **Format:** Save the final video file as `Team_Rescaled_Video_Presentation.mp4` inside the Google Drive folder under `5_Video_Presentation_Script/` (or upload as an unlisted YouTube / Google Drive link).

---

## Timestamped Script & Slide Cues

```
========================================================================================
[0:00 - 0:30] SLIDE 1: Title Slide & Introduction
----------------------------------------------------------------------------------------
```
**[Presenter 1]:**  
> *"Good day, Professor Bhaskarjyoti Das, TAs, and colleagues. Welcome to our presentation for Advanced Data Analytics.  
> 
> Our project is titled: **'Does the Way We Transform Outcomes Change Who We Target? Scale-Dependence of CATE and Uplift Rankings for Zero-Heavy Outcomes, with Count-Aware Learners'**.  
> 
> On behalf of Team Rescaled—comprising Pallavi, Khushi, Lakshanya, and Neha—today we demonstrate a critical, previously unstudied issue in causal machine learning: how standard outcome transformations silently corrupt customer targeting decisions in commercial uplift models, and how count-aware likelihood models resolve this problem."*

```
========================================================================================
[0:30 - 1:05] SLIDE 2: Motivation & The Zero-Heavy Problem
----------------------------------------------------------------------------------------
```
**[Presenter 1]:**  
> *"In real-world marketing and e-commerce trials, spend outcomes exhibit severe zero inflation—typically 90 to 99% exact zeros—paired with heavy right tails.  
> 
> To stabilize variance and satisfy squared-error loss, practitioners universally apply concave transformations like $\log(1 + ay)$ or $\text{asinh}(ay)$. Here, the parameter $a$ represents the reporting unit: $a=1$ for dollars, $a=0.01$ for cents, or $a=1000$ for thousands.  
> 
> The industry treats $a$ as an innocent data pre-processing choice. But our core research question asks: **Does changing the measurement scale $a$ secretly change WHO gets targeted by causal policies, and how much revenue payoff is lost?**"*

```
========================================================================================
[1:05 - 1:45] SLIDE 3: Theoretical Foundation & Asymptotic Limits (Chen & Roth 2024)
----------------------------------------------------------------------------------------
```
**[Presenter 2]:**  
> *"To answer this, we build on the econometrics foundation of Chen and Roth (2024) and extend it to individual-level CATE and uplift rankings.  
> 
> Consider the transformed conditional treatment effect $\tau_{m,a}(x)$. Mathematically, as scale $a$ approaches zero, $\frac{\log(1+ay)}{a}$ converges to $y$. Thus, at small scales (cents or dollars), the model targets **Level Dollars** ($\tau_{\text{level}}$), heavily prioritizing large spenders on the intensive margin.  
> 
> However, as scale $a$ approaches infinity, spend magnitudes are entirely erased, and the estimator converges strictly to **Conversion Probability** ($\tau_{\text{extensive}}$).  
> 
> In reality, marketing promotions persuade low-spend users to convert, but encourage loyal buyers to spend more. Because these two margins diverge across customer segments, changing the reporting scale violently inverts customer targeting priority."*

```
========================================================================================
[1:45 - 2:25] SLIDE 4: Methodology & Architecture
----------------------------------------------------------------------------------------
```
**[Presenter 2]:**  
> *"To study and solve this, we engineered a complete modular causal pipeline with 5-fold cross-fitting across T-Learners, X-Learners, and Doubly Robust (DR) Learners.  
> 
> Rather than relying on arbitrary transformations, we introduce native count-aware likelihood models:  
> 1. Compound Poisson-Gamma **Tweedie** deviance ($p=1.5$), which naturally models discrete mass at zero and continuous spend simultaneously.  
> 2. **Two-Part Hurdle** models combining a binary conversion classifier with a log-gamma regressor.  
> 
> Our entire codebase is verified by an exhaustive suite of **192 automated unit tests** passing with 100% integrity, covering DGP invariants, models, metrics, and oracle benchmarks."*

```
========================================================================================
[2:25 - 3:10] SLIDE 5: Synthetic Benchmarks & Oracle Ranking Analysis
----------------------------------------------------------------------------------------
```
**[Presenter 1]:**  
> *"Turning to our results, Slide 5 highlights our Oracle Ranking Analysis, shown in Figure 1 directly from our repository.  
> 
> By calculating true CATE directly from potential outcome equations, we isolate estimand distortion from model estimation noise. In scenarios where extensive and intensive margins conflict—such as the 'Opposed' and 'Extensive' scenarios—Spearman rank correlation under $\log(1+ay)$ collapses drastically as $a$ changes across zero-rate regimes of 70%, 90%, and 98%.  
> 
> Meanwhile, strictly monotonic power transforms like square-root and our count-aware Tweedie and Hurdle models maintain exact scale-equivariance ($\rho = 1.0000$) across all scales."*

```
========================================================================================
[3:10 - 3:50] SLIDE 6 & 7: Real-World Field Validation (Hillstrom Trial, N=64,000)
----------------------------------------------------------------------------------------
```
**[Presenter 2]:**  
> *"To test practical commercial impact, we evaluated the genuine Hillstrom randomized marketing trial ($N=64{,}000$, 99.1% zeros).  
> 
> First, on Slide 6, we discovered that merely switching reporting scale from dollars ($a=1$) to cents ($a=0.01$) under $\log(1+ay)$ causes **48.2% of the top-20% targeted customers to be replaced**! The model stops targeting high-value buyers and wastes budget on frequent small buyers.  
> 
> On Slide 7, we compute the realized Inverse Propensity Weighted (IPW) policy value. **Tweedie achieves the peak empirical welfare of \$0.8566 per targeted user**, outperforming naive level models (\$0.7302) and log-transformed regression (\$0.7831) by **+\$0.12 per targeted customer**, completely eliminating targeting churn."*

```
========================================================================================
[3:50 - 4:20] SLIDE 8: Conclusions & Practical Decision Framework
----------------------------------------------------------------------------------------
```
**[Presenter 1]:**    
> *"In conclusion, transforming zero-heavy spend outcomes is not an innocent data cleaning step—it fundamentally alters the causal estimand and causes severe revenue bleed.  
> 
> For industry practitioners, we provide a clean decision framework:  
> - If optimizing conversion, model the binary indicator directly.  
> - If optimizing dollar revenue on zero-heavy spend, **never apply $\log(1+ay)$**. Standardize on native Compound Poisson-Gamma Tweedie ($p=1.5$) or Two-Part Hurdle models with cross-fitting.  
> 
> All 6 course deliverables—our Final One-Pager, Master Jupyter Notebook, Datasets, Presentations, Video Script, and IEEE Report—are organized and ready in our Google Drive folder.  
> 
> Thank you, and we welcome your questions!"*

---
*Deliverable 5 of 6 | Team Rescaled | Advanced Data Analytics (ADA-2026)*


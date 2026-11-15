---
version: 1.0.0
name: "ReviewForge"
description: >-
  An authoritative, AI-assisted academic literature review synthesis engine running Inter for structural controls and Plus Jakarta Sans / Source Serif for scholarly prose on a pure-white canvas. The system's signature accent is electric mint green #31d891, used exclusively as a border voltage and verification highlight rather than a solid button fill. Primary actions feature an ink-black #111111 fill with a 2px mint border, while a deep-navy #100841 anchors the focal Corpus & Plagiarism Audit island. Zero box-shadows, 2px hairline borders, and tight 4-5px card radii define an archival, publication-grade academic workstation.

seo:
  title: "ReviewForge Design System — Academic Review Synthesis Engine, Mint Voltage #31d891, Ink CTAs, Dual Typeface"
  metaDescription: "ReviewForge's design system specification — flat-editorial academic aesthetic, electric mint verification signals, ink-black CTAs with 2px borders, 16 color tokens, and publication-ready component definitions."
  highlights:
    - "Border-only brand voltage — electric mint green #31d891 is reserved for 2px button borders, verification indicators, and text marks; never solid button fills"
    - "Dual-typeface semantic register — Inter carries navigation, tabs, inputs, and headings at weight 400; Plus Jakarta Sans / serif handles scholarly abstracts, paper cards, and drafted literature review sections"
    - "Weight-400 heading restraint — scale communicates authority, not heavy weights; display headers run at weight 400 with negative tracking (-1.2px to -2.0px)"
    - "Zero-shadow elevation — flat editorial depth governed entirely by 1.5px–2px hairlines (#dfdfdf) on pure white (#ffffff)"
    - "Deep-navy audit island — #100841 serves as an isolated dark band for the Corpus & Plagiarism Verification block"
    - "Tight 4px card radii — disciplined, table-precision rounding for academic data modules and paper cards"
  tags:
    - "Academic Research"
    - "AI Literature Review"
    - "Scientific Verification"
    - "Design System"

colors:
  primary: "#31d891"
  primary-muted: "#31a679"
  primary-dark: "#31715f"
  primary-link-hover: "#1b4438"
  ink: "#111111"
  canvas: "#ffffff"
  surface-dim: "#100841"
  surface-subtle: "#f9fafb"
  hairline: "#dfdfdf"
  hairline-muted: "#e5e7eb"
  ink-muted: "#4f525d"
  ink-dim: "#868c95"
  placeholder: "#6b707a"
  error: "#df0134"
  error-bg: "#fef2f2"
  warning: "#b45309"
  warning-bg: "#fefce8"
  success: "#15803d"
  success-bg: "#f0fdf4"

typography:
  display-xl:
    fontFamily: "\"Inter\", system-ui, -apple-system, sans-serif"
    fontSize: 44px
    fontWeight: 400
    lineHeight: 52px
    letterSpacing: "-1.8px"
  display-lg:
    fontFamily: "\"Inter\", system-ui, -apple-system, sans-serif"
    fontSize: 32px
    fontWeight: 400
    lineHeight: 40px
    letterSpacing: "-1.2px"
  heading-md:
    fontFamily: "\"Inter\", system-ui, -apple-system, sans-serif"
    fontSize: 22px
    fontWeight: 500
    lineHeight: 28px
    letterSpacing: "-0.6px"
  heading-sm:
    fontFamily: "\"Inter\", system-ui, -apple-system, sans-serif"
    fontSize: 16px
    fontWeight: 600
    lineHeight: 22px
    letterSpacing: "-0.2px"
  eyebrow:
    fontFamily: "\"Inter\", system-ui, -apple-system, sans-serif"
    fontSize: 12px
    fontWeight: 700
    lineHeight: 16px
    letterSpacing: "1.5px"
    textTransform: "uppercase"
  body-lg:
    fontFamily: "\"Plus Jakarta Sans\", \"Georgia\", serif"
    fontSize: 17px
    fontWeight: 400
    lineHeight: 28px
    letterSpacing: "0px"
  body-md:
    fontFamily: "\"Plus Jakarta Sans\", system-ui, sans-serif"
    fontSize: 15px
    fontWeight: 400
    lineHeight: 24px
    letterSpacing: "0px"
  body-sm:
    fontFamily: "\"Plus Jakarta Sans\", system-ui, sans-serif"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 18px
    letterSpacing: "0px"
  mono-code:
    fontFamily: "\"SF Mono\", \"Monaco\", \"Inconsolata\", monospace"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 18px

rounded:
  none: "0px"
  xs: "4px"
  sm: "5px"
  md: "10px"
  lg: "14px"
  pill: "70px"

spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  base: "24px"
  lg: "32px"
  xl: "48px"
  2xl: "64px"

components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.canvas}"
    borderColor: "{colors.primary}"
    borderWidth: "2px"
    rounded: "{rounded.pill}"
    padding: "10px 24px"
    typography: "{typography.heading-sm}"
  button-primary-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.primary}"
    borderColor: "{colors.ink}"
    borderWidth: "2px"
    rounded: "{rounded.pill}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    borderColor: "{colors.hairline}"
    borderWidth: "1.5px"
    rounded: "{rounded.pill}"
    padding: "8px 20px"
  paper-card:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    borderColor: "{colors.hairline}"
    borderWidth: "1.5px"
    rounded: "{rounded.md}"
    padding: "20px"
  audit-island:
    backgroundColor: "{colors.surface-dim}"
    textColor: "{colors.canvas}"
    borderColor: "transparent"
    rounded: "{rounded.md}"
    padding: "28px 32px"
  evidence-callout:
    backgroundColor: "{colors.surface-subtle}"
    textColor: "{colors.ink}"
    borderLeftColor: "{colors.primary}"
    borderLeftWidth: "3px"
    rounded: "{rounded.xs}"
    padding: "10px 14px"
  text-input:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    borderColor: "{colors.hairline}"
    borderWidth: "1.5px"
    rounded: "{rounded.xs}"
    padding: "8px 12px"
---

# ReviewForge Design System

## Overview
ReviewForge is an academic literature synthesis and verification workstation. Its design philosophy translates FinScope AI's editorial discipline into the academic domain: replacing the colorful, noisy aesthetics of generic AI wrappers with an **authoritative, archival, preprint-grade interface**.

## 1. Color System

### Brand Voltage (Electric Mint Green)
- **Primary** (`#31d891`): The electric voltage signal. Used strictly on **2px button borders**, verification checkmarks, active tab indicators, and inline quote markers. **Never** used as a solid button fill.
- **Primary Muted** (`#31a679`): Secondary interactive states, subtle progress rings, and active state outlines.
- **Primary Dark** (`#31715f`): Scholarly citation links (`[1]`, `[2]`), DOI anchors, and verified badge text.

### Ink & Canvas
- **Ink** (`#111111`): Deep carbon black used for primary typography, primary button fills, and high-emphasis borders.
- **Canvas** (`#ffffff`): Pure white ground plane for paper preview sheets and work surfaces.
- **Surface Subtle** (`#f9fafb`): Light grey tint for evidence callouts, code containers, and table headers.

### The Focal Island
- **Surface Dim** (`#100841`): A rich, deep navy dark band isolated specifically to the **Corpus & Plagiarism Audit** block and summary analytics. It serves as the single high-contrast tonal anchor on the interface.

### Hairlines & Mutes
- **Hairline** (`#dfdfdf`): Crisp 1.5px–2px borders on cards, inputs, and section boundaries.
- **Hairline Muted** (`#e5e7eb`): Soft interior table grid lines and divider rules.
- **Ink Muted** (`#4f525d`): Author listings, venues, and secondary metadata.
- **Ink Dim** (`#868c95`): Captions, timestamps, and input placeholders.

### Scientific Verification & Status
- **Success / Clean** (`#15803d` / `#f0fdf4`): Plagiarism overlap $\le 15\%$ (Green Band).
- **Warning / Review** (`#b45309` / `#fefce8`): Plagiarism overlap $15\% - 25\%$ (Yellow Band).
- **Error / High Overlap** (`#df0134` / `#fef2f2`): Overlap $> 25\%$ (Red Band) or API failure.

---

## 2. Typography: Semantic Register Split

ReviewForge avoids arbitrary font sizing by enforcing two semantic registers:

1. **Structural & Navigational (Inter)**:
   - Covers UI controls, navigation tabs, section headers, inputs, and eyebrow tags.
   - Headers use **Weight 400** with tight negative tracking (`-1.2px` to `-1.8px`), communicating calm confidence rather than loud marketing bolding.
   - Eyebrows run at **Weight 700**, `12px`, uppercase with `+1.5px` letter spacing.

2. **Editorial & Scholarly Prose (Plus Jakarta Sans / Source Serif)**:
   - Covers paper abstracts, synthesized literature review sections, evidence quotations, and research questions.
   - Generous line-height (`1.65` to `1.75`) optimized for rigorous multi-paragraph scientific synthesis.

---

## 3. Elevation & Shape

- **No Drop Shadows**: All visual elevation and boundaries are achieved via clean `1.5px` to `2px` hairline borders (`#dfdfdf`).
- **Disciplined Radii**:
  - `4px` (`xs`): Code blocks, evidence quotes, input fields.
  - `5px` (`sm`): Badges, citation chips.
  - `10px` (`md`): Paper Cards, Analysis Cards, Tab Panels.
  - `70px` (`pill`): Primary & Secondary CTA buttons.

---

## 4. Component Rules

- **Primary Button**: Background `#111111`, text `#ffffff`, border `2px solid #31d891`, radius `70px`. On hover, background stays `#111111`, border transitions to `#111111`, text becomes `#31d891`.
- **Secondary Button**: Background `transparent`, text `#111111`, border `1.5px solid #dfdfdf`, radius `70px`. On hover, border transitions to `#31d891`.
- **Paper Card**: White background, `1.5px solid #dfdfdf` border, `10px` radius, `20px` internal padding.
- **Evidence Callout**: `#f9fafb` background, `3px solid #31d891` left accent border, `4px` radius.
- **Audit Island**: Deep navy `#100841` background, `#ffffff` text, `10px` radius, `28px` padding.

---

## 5. Iconography: Zero-Emoji Policy

To maintain a publication-grade, institutional workstation aesthetic, ReviewForge adheres to a **Strict Zero-Emoji Policy**:
- **No Consumer Emojis**: Cartoon emojis (e.g., 🚀, 🔬, 📄, 💡, 🛡️, 📥) are strictly forbidden across buttons, tabs, labels, and status dialogs.
- **Official Material Symbols**: All interactive actions, navigation tabs, and system indicators use Google Material Symbols Outlined (`:material/<icon_name>:` in Streamlit).
- **Semantics & Restraint**: Icons serve as clear affordances, not decorative noise.

---

## 6. Do's and Don'ts

- **DO** keep primary button backgrounds ink-black (`#111111`) with a 2px mint border (`#31d891`).
- **DO** use regular weight (400) with negative tracking for major headings.
- **DO** confine the deep-navy `#100841` surface to the audit block and high-emphasis summaries.
- **DO** use official `:material/<icon_name>:` vector icons for tabs, buttons, and status cues.
- **DON'T** use casual or consumer emojis anywhere in the interface.
- **DON'T** use mint green as a solid button fill.
- **DON'T** apply drop shadows to cards or modals.
- **DON'T** use puffy 16px+ border radii on content containers.

---
name: VACV SI Workbench
description: A local molecular research workbench with pale sage surfaces and precise specimen captions.
colors:
  paper: "#f5f5ef"
  white: "#fff"
  ink: "#173f38"
  text: "#24483f"
  muted: "#56645d"
  line: "#d6ddd4"
  accent: "#176d59"
  ochre: "#926227"
  error: "#a5342b"
  soft: "#e8eee5"
  accent-hover: "#125542"
  field-border: "#aab8ab"
  result-paper: "#fbfcf8"
  error-paper: "#fff0e9"
  selected-row: "#e5ede2"
typography:
  display:
    fontFamily: "Golos, sans-serif"
    fontSize: "clamp(28px, 3.1vw, 43px)"
    fontWeight: 580
    lineHeight: 1.17
    letterSpacing: "-.035em"
  headline:
    fontFamily: "Golos, sans-serif"
    fontSize: "18px"
    fontWeight: 580
    lineHeight: 1.35
  title:
    fontFamily: "Golos, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.6
  body:
    fontFamily: "Golos, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Golos, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.6
  control:
    fontFamily: "Golos, sans-serif"
    fontSize: "13px"
    fontWeight: 550
    lineHeight: 1.6
  structure:
    fontFamily: "Consolas, monospace"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.6
  estimate:
    fontFamily: "Golos, sans-serif"
    fontSize: "36px"
    fontWeight: 550
    lineHeight: 1.25
    letterSpacing: "-.03em"
rounded:
  control: "6px"
  container: "12px"
spacing:
  "4": "4px"
  "8": "8px"
  "12": "12px"
  "16": "16px"
  "20": "20px"
  "24": "24px"
  "28": "28px"
  "30": "30px"
  "32": "32px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.white}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "10px 18px"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
  button-secondary:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "10px 18px"
  button-secondary-hover:
    backgroundColor: "{colors.soft}"
  structure-input:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    typography: "{typography.structure}"
    rounded: "{rounded.control}"
    padding: "14px"
  workspace:
    backgroundColor: "{colors.white}"
    rounded: "{rounded.container}"
  result-surface:
    backgroundColor: "{colors.result-paper}"
    padding: "28px 30px 26px"
  decision:
    backgroundColor: "{colors.soft}"
    rounded: "{rounded.container}"
    padding: "24px 28px"
  error:
    backgroundColor: "{colors.error-paper}"
    textColor: "{colors.error}"
    rounded: "{rounded.control}"
    padding: "12px 14px"
---

# Design System: VACV SI Workbench

## Overview

**Creative North Star: "The Botanical Specimen Collection"**

The botanical specimen collection is expressed through pale sage paper, dark green typography, economical rules, and precise captions. The molecular drawing is the visual subject; surrounding controls stay quiet and legible.

This is a compact research workbench. The interface pairs an editable structure with a readable result, then makes the validation evidence available through the same navigation. Russian copy uses self-hosted Golos Text; tabular numerals keep scientific values stable. Status, error, and model limitations remain written information.

**Key Characteristics:**

- Pale sage paper with white working surfaces and a subtly tinted result surface.
- One green action accent; ochre focus outlines; explicit textual states.
- Compact variable-weight typography and tabular scientific values.
- Flat, lightly ruled containers; molecular drawings and scientific plots carry the imagery.

This record describes the completed build in `web/app.css`, `web/index.html`, and `web/app.js`. The direction in `docs/ui-direction.md` provides the metaphor; the final CSS cascade provides the values.

## Colors

A restrained green palette places dark specimen text on pale paper and white working surfaces. Frontmatter values are normative.

### Primary

- **Specimen Green** (`accent`): primary actions, links, active-navigation underline, readiness dot, and the input caret.
- **Deep Action Green** (`accent-hover`): primary-control hover state.
- **Sage Wash** (`soft`): decision summaries and secondary-control hover.
- **Selected Sage** (`selected-row`): the table row for the currently used model.

### Secondary

- **Evidence Ochre** (`ochre`): visible keyboard focus. The plot captions also identify ochre as the special-label plot color; plot styling remains owned by the scientific figure artifacts.
- **Error Red** (`error`) and **Error Paper** (`error-paper`): textual error notices, always accompanied by a message.

### Neutral

- **Sage Paper** (`paper`): page canvas.
- **White Work Surface** (`white`): header, entry region, inputs, and secondary controls.
- **Result Paper** (`result-paper`): slightly tinted structure/result region.
- **Deep Green Ink** (`ink`): headings, important values, and input content.
- **Body Green** (`text`): default reading text.
- **Caption Grey-Green** (`muted`): contextual copy, inactive navigation, and supporting labels.
- **Pale Rule** (`line`): regional dividers, table rules, and secondary-control borders.
- **Field Stroke** (`field-border`): stronger input boundary.

**The Green Action Rule.** Use the green accent for actionable controls, links, selected navigation, and readiness. Preserve written status alongside color.

The sidecar's synthesized tonal strips are panel previews derived from these colors, not additional approved application tokens.

## Typography

**Display Font:** Golos Text, registered as `Golos`, with sans-serif fallback.  
**Body Font:** the same self-hosted variable family.  
**Label/Mono Font:** Golos for captions; Consolas with monospace fallback for editable SMILES.

**Character:** Close-set page headings provide hierarchy without a separate display face. Compact captions and tabular numerals support scientific comparison. The font is loaded locally from `/fonts/GolosText.ttf` with `font-display: swap`.

### Hierarchy

- **Display:** page heading; the frontmatter fluid size applies above the narrow breakpoint. At the narrow breakpoint it becomes (31px), with the same weight, tracking, and line height.
- **Headline:** section heading. Workbench section headings become (16px) on narrow screens; the comparison-table heading becomes (17px).
- **Title:** short context headings.
- **Body:** introductory reading text. Quality-analysis paragraphs use (14px); narrow-screen introduction copy also uses (14px).
- **Label:** supporting status, helper copy, table headings, and captions. Form labels use (13px) with weight (600).
- **Control:** primary and secondary actions.
- **Structure:** editable molecular notation; the placeholder returns to Golos at (13px).
- **Estimate:** the principal SI value; narrow screens use (31px). Similarity uses (26px), including on narrow screens because its more specific selector wins.

All page text inherits `font-variant-numeric: tabular-nums`. Supporting result and limitation paragraphs use a slightly more open line height (1.7). The limitation note caps its line length at (65ch).

**The Measured Hierarchy Rule.** Keep the main estimate visually stronger than similarity, and retain labels and explanatory text beside both.

## Layout

The centered main and footer containers cap at (1344px), including horizontal gutters of (32px). The desktop header is (84px) tall; its content aligns to the (1280px) interior. Main content begins with (48px) top padding.

The main work surface uses a two-column grid (1fr / 1.05fr). Input and explanatory context occupy the left column; the result spans both rows on the right. The initial result has a centered empty state. Default work-region padding is (28px 30px 26px); the final cascade removes the input's bottom padding and gives the context its own (0 30px 26px) padding. The result is separated with a vertical rule.

At (900px) and below, main gutters become (24px) and work-region padding contracts to (23px). The local-computer header label disappears. The experiment/run status remains visible after the final CSS override.

At (650px) and below, navigation moves onto its own row; main gutters become (16px). The work surface becomes one column in DOM order: input, result, then explanation and neighbors. Region padding becomes (23px 20px), and horizontal rules replace the column divider. Action buttons share the row and may wrap through the flex container. The experiment/run status remains beneath the introduction.

The quality view uses a full-width comparison table and a two-column findings region that becomes one column on narrow screens. Numerical columns align right. The narrow comparison table retains a minimum width of (560px); scientific plots retain a minimum width of (700px). Both use labeled, keyboard-focusable horizontal scroll regions, with mobile instructions and full-image links for plots. The wide plot is intentionally inspectable rather than shrunk into unreadability.

The spacing vocabulary is compact and varied; it is not a strict single-step grid. Use the extracted reused values, with wider gaps reserved for major regions.

## Elevation & Depth

Depth comes from white and lightly tinted surfaces, pale rules, and spacing. The shipped interface has no box shadows. Focus is an outline, not elevation.

**Thed Surface Rule Rule.** Separate working regions with pale borders and surface tones. The shipped interface has no box shadows.

Interactive buttons use a background transition of (.15s). Scrolling is smooth by default. The reduced-motion media query disables transitions and restores automatic scrolling.

## Shapes

Containers have gently rounded corners through the container token; controls and error messages use the smaller control token. Rules and input borders are (1px). The active navigation underline is (2px). The readiness indicator is a small circle (8px), paired with a textual status.

The molecular logo and empty-state drawing use inline outline SVG. Real molecular imagery uses the generated structure drawing; scientific plots are separate SVG artifacts. Neither the specimen metaphor nor the rounded container implies decorative ornament.

## Components

### Buttons

Compact, explicit actions with a minimum height of (44px). Primary uses Specimen Green with white text; secondary uses a white fill, Deep Green Ink, and a Pale Rule border. Their shared geometry, typography, and padding are frontmatter tokens. Primary hover deepens the green; secondary hover adds Sage Wash.

Keyboard focus uses an ochre outline (3px), offset (4px). Disabled buttons have opacity (.65) and a waiting cursor. During calculation, both actions are disabled and the SMILES field is read-only. Arrow imagery is inline SVG, not a text glyph.

### Cards / Containers

The main work surface is a white, ruled, rounded container with clipped overflow; its result region has Result Paper. The validation decision uses a Sage Wash container. These containers do not lift on hover.

### Inputs / Fields

The SMILES field is a full-width resizable textarea with a Field Stroke border, white background, mono content, and the control radius. Its height ranges from (132px) to (320px). Long notation wraps within the field. Focus uses a green outline (2px), offset (2px).

The field has a persistent visible label, linked helper/error copy, and programmatic invalid state. Error messages use the Error Paper container and Error Red text. Editing or loading another structure clears the displayed prediction until recalculated.

### Navigation

A white header contains the molecular wordmark, two textual navigation links, and desktop local-context text. Navigation uses muted (14px) text; the current page uses green ink, weight (600), and a green bottom border. Hover changes link color to the accent. On narrow screens links use (13px), sit on a second row, and retain both text and active underline.

### Structure and Result

The principal signature is the real molecular structure above a compact pair of labeled values. The image is contained at a fixed display height (235px), becoming (205px) on narrow screens. SI is larger than training similarity. A pale divider separates the image from the values.

Before a calculation, an outline molecule and a short instruction occupy the result region. The empty state has minimum height (328px), becoming (235px) on narrow screens. During calculation the result region uses opacity (.65) and an explicit status label. The region announces updates politely. Canonical SMILES is revealed through a native disclosure.

### Tables and Scientific Figures

Tables use horizontal rules without vertical grid lines. Row identifiers and strategy labels align left; values align right. The used strategy has a Selected Sage background and written “используется” text. The full search procedure is green text. Supporting neighbor tables use smaller type; quality tables use (14px) body text and larger cell padding (15px 14px).

Plot images carry descriptive alternative text and captions explaining their scientific scope. Preserve the scroll and full-image access described in Layout.

## Do's and Don'ts

### Do:

- **Do** use the actual molecular drawing and scientific plots as the primary imagery.
- **Do** retain written state labels, visible keyboard focus, and the reduced-motion override.
- **Do** keep numerical table columns right aligned and use tabular numerals.
- **Do** preserve input, result, then explanatory context in the narrow-screen reading order.
- **Do** provide keyboard-accessible horizontal scrolling and a full-image link for wide plots.

### Don't:

- **Don't** replace scientific evidence with decorative science imagery.
- **Don't** promote similarity into a confidence probability or make a model estimate look like a measurement.
- **Don't** add decorative shadows to the flat ruled surfaces.
- **Don't** convey active navigation, errors, or model readiness through color alone.

Not canonized: the unused row-tag selector, one-off decorative colors, and earlier overridden layout declarations are not reusable system rules. This documentation pass did not change or repair the application.


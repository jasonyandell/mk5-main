# PDF Form Filling Guide

This document provides instructions for completing PDF forms using automated scripts. The process differs based on whether the PDF contains fillable form fields.

## Initial Step: Check for Fillable Fields

Before proceeding, run `python scripts/check_fillable_fields.py <file.pdf>` to determine which workflow applies to your document.

## Fillable Fields Workflow

For PDFs with built-in form fields:

1. **Extract field information** using `python scripts/extract_form_field_info.py <input.pdf> <field_info.json>`

2. **Convert to images** with `python scripts/convert_pdf_to_images.py <file.pdf> <output_directory>` to visually verify field purposes

3. **Create field values file** in `field_values.json` format, matching field IDs with appropriate values:

```json
[
  {
    "field_id": "name",
    "page": 1,
    "value": "John Doe"
  },
  {
    "field_id": "date",
    "page": 1,
    "value": "2024-01-15"
  }
]
```

4. **Fill the PDF** using `python scripts/fill_fillable_fields.py <input pdf> <field_values.json> <output pdf>`

## Non-Fillable Fields Workflow

For PDFs without form fields, follow four sequential steps:

### Step 1: Visual Analysis

Convert PDF to PNG images and identify all form fields and data entry areas. Determine bounding boxes for both labels and entry zones.

**CRITICAL**: Label and entry bounding boxes MUST NOT INTERSECT. Entry areas should only cover the space where user input goes.

```bash
python scripts/convert_pdf_to_images.py <file.pdf> <output_directory>
```

### Step 2: Create Configuration and Validation Images

Generate a `fields.json` file containing page dimensions and field specifications:

```json
{
  "pages": [
    {
      "page_number": 1,
      "image_width": 850,
      "image_height": 1100
    }
  ],
  "form_fields": [
    {
      "page_number": 1,
      "description": "Full Name",
      "label_bounding_box": [50, 100, 150, 120],
      "entry_bounding_box": [160, 100, 400, 120],
      "entry_text": {
        "text": "John Doe",
        "font": "Arial",
        "font_size": 12,
        "font_color": "000000"
      }
    }
  ]
}
```

Bounding box format: `[left, top, right, bottom]` in image pixel coordinates.

Create validation images showing rectangles:
```bash
python scripts/create_validation_image.py <page_number> <fields.json> <input_image> <output_image>
```

### Step 3: Validate Bounding Boxes

Run automated intersection check:
```bash
python scripts/check_bounding_boxes.py <fields.json>
```

Manually inspect validation images:
- **Red rectangles** = entry areas (should cover only input space, no text)
- **Blue rectangles** = labels (should contain label text properly)

### Step 4: Add Annotations

Generate the completed PDF:
```bash
python scripts/fill_pdf_form_with_annotations.py <input_pdf> <fields.json> <output_pdf>
```

## Field Types Reference

### Text Fields
```json
{
  "field_id": "name",
  "type": "text",
  "page": 1,
  "rect": [100, 200, 300, 220]
}
```

### Checkboxes
```json
{
  "field_id": "agree",
  "type": "checkbox",
  "page": 1,
  "checked_value": "/Yes",
  "unchecked_value": "/Off"
}
```

### Radio Groups
```json
{
  "field_id": "gender",
  "type": "radio_group",
  "page": 1,
  "radio_options": [
    {"value": "/Male", "rect": [100, 200, 120, 220]},
    {"value": "/Female", "rect": [150, 200, 170, 220]}
  ]
}
```

### Choice/Dropdown
```json
{
  "field_id": "country",
  "type": "choice",
  "page": 1,
  "choice_options": [
    {"value": "US", "text": "United States"},
    {"value": "CA", "text": "Canada"}
  ]
}
```

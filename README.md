# Pascal Brokmeier's CV

This repository contains my professional CV built using [RenderCV](https://rendercv.com), a
Typst-based CV framework designed for academics and engineers.

## Features

### Custom Theme: `modcvcustomlong`

I've created a custom theme that extends the moderncv design with the following features:

- **Long-form CV format**: Designed for comprehensive content without page restrictions
- **Clickable online resources**: Links in the online resources section are clickable when URLs are
  provided
- **Custom typography**: Uses Fontin font family for a clean, professional look
- **Modern layout**: Clean section headers with horizontal lines
- **ATS-friendly**: Optimized for Applicant Tracking Systems while maintaining visual appeal

### Key Sections

- **Summary**: High-level overview with quantifiable achievements
- **Work Experience**: Detailed role descriptions with metrics and impact
- **Education**: Academic background with research highlights
- **Skills**: Technical competencies organized by domain
- **Awards**: Recognition and achievements
- **Online Resources**: Clickable links to blog, podcast, and presentations

## Usage

### Prerequisites

- Python 3.8+
- RenderCV

### Installation

```bash
# using uv (recommended)
uv sync
```

### Rendering

```bash
# Generate all formats
uv run rendercv render Pascal_Brokmeier_CV.yaml

# Generate specific format
uv run rendercv render Pascal_Brokmeier_CV.yaml --format pdf
```

### Output Formats

The CV is generated in multiple formats:

- PDF (primary)
- HTML (web version)
- Markdown (text version)
- Typst (source)

## Custom Theme Details

The `modcvcustomlong` theme includes:

- **Header**: Clean layout with contact information and social links
- **Sections**: Modern headers with horizontal lines
- **Entries**: Two-column layout with dates/locations on the right
- **Links**: Underlined links with optional external link icons
- **Typography**: Fontin font family throughout

### Template Files

- `NormalEntry.j2.typ`: Handles online resources with clickable links
- `ExperienceEntry.j2.typ`: Work experience formatting
- `EducationEntry.j2.typ`: Education section layout
- `Header.j2.typ`: Header with contact information
- `Preamble.j2.typ`: Theme configuration and styling

## Content Strategy

The CV follows a strategic approach:

1. **Top messages**: Key achievements and positioning at the beginning
2. **ATS optimization**: Sufficient keywords and structured content
3. **Quantified achievements**: Specific metrics and impact numbers
4. **Global Fortune 500 experience**: Highlighting enterprise-level work
5. **Technical depth**: Demonstrating hands-on engineering leadership

## Updates

The CV is regularly updated to reflect:

- New achievements and projects
- Updated metrics and impact numbers
- Current positioning and career goals
- Latest online content and publications

## License

This CV template is based on RenderCV and is available under the same license terms.

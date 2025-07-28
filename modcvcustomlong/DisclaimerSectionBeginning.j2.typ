#v(0.2em)
#align(left, text(
  size: 0.7em,
  fill: grey,
  style: "italic",
  weight: "normal",
  "Disclaimer"
))
#v(0.1em)
#line(length: 100%, stroke: 0.3pt + grey)
#v(0.2em)
((* if not design.entries.allow_page_break_in_sections *))
#block(
  [
((* endif *))
((* if entry_type in ["NumberedEntry", "ReversedNumberedEntry"] *))
#one-col-entry(
  content: [
  ((* if entry_type == "ReversedNumberedEntry" *))
    #let rev-enum-items = (
  ((* endif *))
((* endif *)) 
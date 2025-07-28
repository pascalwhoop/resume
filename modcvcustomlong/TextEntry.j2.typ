((* if section_title|lower == "disclaimer" *))
#one-col-entry(
  content: [
    #text(
      size: 0.9em,
      fill: rgb(128, 128, 128),
      style: "italic",
      weight: 400,
      [Disclaimer:  <<entry>>]
    )
  ]
)
((* else *))
#one-col-entry(
  content: [<<entry>>]
)
((* endif *))
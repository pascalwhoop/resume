yaml = require('js-yaml');
fs = require('fs');
// load resume.yaml
let resumeYaml = fs.readFileSync('resume.yaml', 'utf8');
obj = yaml.load(resumeYaml)
// write out as resume.json
fs.writeFileSync('resume.json', JSON.stringify(obj, null, 2), 'utf8');
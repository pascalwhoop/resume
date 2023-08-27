const themeServer = process.env.THEME_SERVER || 'https://themes.jsonresume.org/theme/';
const fs = require('fs');
const request = require('superagent');
const chalk = require('chalk');
const renderHtml = require('../render-html');

const denormalizeTheme = (value) => {
  return value.match(/jsonresume-theme-(.*)/)[1];
};

const sendExportHTML = (resumeJson, theme, callback) => {
  console.log(resumeJson, theme);
  console.log('Requesting theme from server...');

  request
    .post(themeServer + denormalizeTheme(theme))
    .send({
      resume: resumeJson,
    })
    .set('Accept', 'application/json')
    .end((err, response) => {
      if (err) {
        callback(
          'There was an error downloading your generated html resume from our server: ' +
            err,
        );
      } else if (response.text) {
        callback(null, response.text);
      } else {
        callback(
          'There was an error downloading your generated html resume from our server.',
        );
      }
    });
};
module.exports = function resumeBuilder(theme, dir, resumeFilename, cb) {
  fs.readFile(resumeFilename, async (err, resumeJson) => {
    if (err) {
      console.log(chalk.yellow('Could not find:'), resumeFilename);
      console.log(
        chalk.cyan('Using example resume.json from resume-schema instead...'),
      );
      resumeJson = require('resume-schema').resumeJson;
    } else {
      try {
        // todo: test resume schema
        // patch make it also work with yaml files
        // let's first check if the file is yaml or json. if json act as before if yaml, first conver to json 
        // and then act as before
        if(resumeFilename.endsWith('.yaml') || resumeFilename.endsWith('.yml')) {
          console.log("resume is yaml, converting")
          const yaml = require('js-yaml');
          resumeJson = yaml.load(resumeJson);
        }
        else{
        resumeJson = JSON.parse(resumeJson);
        }
      } catch (e) {
        err = 'Parse error: ' + resumeFilename;
        console.log(e)
        return cb(err);
      }
    }

    try {
      const html = await renderHtml(resumeJson, theme);
      cb(null, html);
    } catch (err) {
      console.log(err);
      console.log(
        chalk.yellow('Could not run the render function from local theme.'),
      );
      sendExportHTML(resumeJson, theme, cb);
    }
  });
};

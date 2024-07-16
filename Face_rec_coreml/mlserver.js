const express = require("express");
var fs = require("fs");
var fse = require("fs-extra");
const path = require("path");
var cors = require("cors");
const { spawn } = require("child_process");
const bodyParser = require("body-parser");
const { PythonShell } = require("python-shell");

const app = express();
app.use(cors());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(express.json({ limit: 10485760 }));

app.post("/upload", (req, res) => {
  console.log("hii from upload Json");
  var key = req.body.uuid;
  var name = req.body.name;
  var imgb64 = req.body.data;
  var count = req.body.count;
  console.log(key, name, count);

  const buffer = Buffer.from(imgb64, "base64");
  var newPath =
    path.join(__dirname, "uploads") +
    "/" +
    key +
    "/dataset/train/" +
    name +
    "/";
  //creating directory if empty
  if (!fs.existsSync(newPath)) {
    fs.mkdirSync(newPath, { recursive: true });
  }
  newPath = newPath + count + ".jpeg";
  //writing images uploaded to fs
  fs.writeFileSync(newPath, buffer);
  res.send(count);
});

app.post("/delete", (req, res) => {
  var key = req.body.uuid;
  var name = req.body.name;
  var ukpath = "uploads/" + key + "/dataset/train" + "/" + name;
  //reuse directory by deleting oldest face when no of faces>5
  xdir = "uploads/" + key + "/dataset/train";

  if (fs.existsSync(ukpath)) {
    dirtodie = "uploads/" + key + "/" + name;
    try {
      fs.rmdirSync(dirtodie, { recursive: true });

      console.log(`${dirtodie} is deleted!`);
    } catch (err) {
      console.error(`Error while deleting ${dirtodie}.`);
    }
    res.write("Deleted");
  } else {
    res.write("Face not found");
  }
  res.end();
});
function getDirectoryNames(uuid) {
  return new Promise((resolve, reject) => {
    const directoryPath = path.join(
      __dirname,
      "uploads/" + uuid + "/dataset/train"
    );
    fs.readdir(directoryPath, { withFileTypes: true }, (err, files) => {
      if (err) {
        console.log("Unable to scan directory: " + err);
        reject("Unable to scan directory");
        return;
      }
      const dirNames = files
        .filter((dirent) => dirent.isDirectory())
        .map((dirent) => dirent.name);
      console.log(dirNames);
      resolve(dirNames);
    });
  });
}
//route for getting list of faces
app.post("/faces", (req, res) => {
  getDirectoryNames(req.body.uuid)
    .then((dirNames) => {
      res.send(dirNames);
    })
    .catch((error) => {
      res.send(error);
    });
});
app.post("/train", (req, res) => {
  console.log("train called");
  var ukpath = "uploads/" + req.body.uuid + "/dataset/train/unknown";
  //reuse directory by deleting oldest face when no of faces>5
  xdir = "uploads/" + req.body.uuid + "/dataset/train";
  fs.readdir(xdir, function (err, files) {
    sortedfiles = files
      .map(function (fileName) {
        return {
          name: fileName,
          time: fs.statSync(xdir + "/" + fileName).mtime.getTime(),
        };
      })
      .sort(function (a, b) {
        return a.time - b.time;
      })
      .map(function (v) {
        return v.name;
      });
    if (files.length > 7) {
      dirtodie = "uploads/" + req.body.uuid + "/" + sortedfiles[0];
      try {
        fs.rmdirSync(dirtodie, { recursive: true });

        console.log(`${dirtodie} is deleted!`);
      } catch (err) {
        console.error(`Error while deleting ${dirtodie}.`);
      }
    }
  });

  //copy unknown files to each uuid
  if (!fs.existsSync(ukpath)) {
    fs.mkdirSync(ukpath, { recursive: true });
    fse.copy("uploads/datset/unknown/", ukpath, (err) => {
      if (err) {
        console.error(err);
      } else {
        console.log("success!");
      }
    });
  }
  getDirectoryNames(req.body.uuid)
    .then((dirNames) => {
      // res.send(dirNames);
      // res.send("Training started");
      let options = {
        args: ["--relative_path", `${req.body.uuid}`, "--classes", ...dirNames],
      };
      console.log("options", options);
      let pyshell = new PythonShell("train.py", options);

      // Listen for messages from the Python script
      pyshell.on("message", function (message) {
        console.log("Message from Python script:", message);
      });

      // Listen for errors
      pyshell.on("stderr", function (stderr) {
        console.error("Error from Python script:", stderr);
      });

      // Listen for the end of the Python script
      pyshell.end(function (err, code, signal) {
        if (err) throw err;
        console.log("Python script finished with code:", code);
        console.log("Python script finished with signal:", signal);
        var fileName = "/app/uploads/" + req.body.uuid + "/people.mlmodel";
        res.sendFile(fileName, options, function (err) {
          if (err) {
            console.log("error while sending");
            console.log(err);
          } else {
            console.log("Sent:", fileName);
          }
        });
        console.log("Closed");
      });
    })
    .catch((error) => {
      res.send(error);
    });
});
app.get("/", (req, res) => {
  res.send("up and running");
});

app.listen(3055, function (err) {
  if (err) console.log(err);
  console.log("Server listening on Port 3055");
});

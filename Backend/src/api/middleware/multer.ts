import  multer from "multer";
import path from "path";

// export default multer({
//   storage: multer.diskStorage({}),
//   fileFilter: (req, file, cb) => {
//     const ext = path.extname(file.originalname);
//     if (ext !== ".jpg" && ext !== ".jpeg" && ext !== ".png") {
//       new Error("File type is not supported")
//       return;
//     }
//     cb(null, true);
//   },
// });

const storage = multer.diskStorage({
  // destination: (req, file, cb) => {
  //   cb(null, "uploads");
  // },
  filename: (req, file, cb) => {
    cb(null, new Date().toISOString().replace(/:/g, '-') + '-' + file.originalname);
  }
});

const fileFilter = (req, file,cb)=>{
  if(file.mimetype === 'image/jpeg' || file.mimetype === 'image/png'|| file.mimetype === 'image/jpg'){
    cb(null,true);
  }else{
    cb(null,false);
  }
}

const uploads = multer({storage: storage,fileFilter:fileFilter});

export {uploads}

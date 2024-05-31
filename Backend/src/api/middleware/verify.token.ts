import jwt from "jsonwebtoken";
const { ACCESS_TOKEN_SECRET } = process.env;

const verify = (req, res, next) => {
  const auth = req.headers.authorization;
  !auth ? null : auth;
  const token = auth?.split(" ")[1];

  if (!token) {
    return res.status(403).send({
      status: "error",
      message: "No token provided!",
    });
  }
  jwt.verify(token, "swsh23hjddnknoh788778aCHOssc", (err, decoded) => {
    if (err) {
      return res.status(401).send({
        status: "error",
        message: "Session expired, Login to continue!",
      });
    }
    req.userId = decoded._id;
   
    next();
  });
};

export { verify}

import express from "express";
const router = express.Router();

import { register, login, sendResetCode, resetAccount, setRole, updateUser, get_user, delete_user } from "../../controllers/user.controller";
import { verify } from "../../middleware/verify.token";

router.post("/api/v1/signup", register);
router.post("/api/v1/signin", login);
router.post("/api/v1/send-reset-code", sendResetCode);
router.post("/api/v1/reset-account", resetAccount);
router.put('/api/v1/set-role', verify, setRole);
router.put('/api/v1/edit-user', verify, updateUser);
router.get('/api/v1/get-user', verify, get_user);
router.delete('/api/v1/delete-user', verify, delete_user);

export default router;

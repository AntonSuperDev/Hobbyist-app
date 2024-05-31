import express from "express";
const app = express();

import AuthRoutes from "../routes/auth/auth.routes";
import ItemRoute from "../routes/items/items.routes";

app.use(AuthRoutes);
app.use(ItemRoute);

app.get("/", async (req, res) => {
  res.send("Hobbyist is running");
});

export default app;

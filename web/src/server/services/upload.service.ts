import { writeFile, mkdir } from "fs/promises";
import path from "path";

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export async function saveUploadedFile(
  file: File,
  subdir: "avatars" | "covers",
): Promise<string> {
  const ext = path.extname(file.name) || ".png";
  const filename = `${generateId()}${ext}`;
  const dir = path.join(process.cwd(), "uploads", subdir);
  await mkdir(dir, { recursive: true });
  const filePath = path.join(dir, filename);
  const buffer = Buffer.from(await file.arrayBuffer());
  await writeFile(filePath, buffer);
  return `/uploads/${subdir}/${filename}`;
}

export async function saveBufferAsFile(
  buffer: Buffer,
  subdir: "avatars" | "covers",
  ext = ".png",
): Promise<string> {
  const filename = `${generateId()}${ext}`;
  const dir = path.join(process.cwd(), "uploads", subdir);
  await mkdir(dir, { recursive: true });
  const filePath = path.join(dir, filename);
  await writeFile(filePath, buffer);
  return `/uploads/${subdir}/${filename}`;
}

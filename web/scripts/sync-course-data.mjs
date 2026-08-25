import { cp, mkdir, rm } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const webDirectory = resolve(scriptDirectory, '..')
const sourceDirectory = resolve(webDirectory, 'data')
const targetDirectory = resolve(webDirectory, 'docs/public/data')

// 训练脚本继续写入原来的 web/data；构建网页前再复制，避免训练层依赖前端目录结构。
await rm(targetDirectory, { recursive: true, force: true })
await mkdir(targetDirectory, { recursive: true })
await cp(sourceDirectory, targetDirectory, { recursive: true })

console.log(`课程数据已同步：${sourceDirectory} -> ${targetDirectory}`)

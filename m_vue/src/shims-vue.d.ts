/* eslint-disable */   // 在整个文件中取消eslint检查
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}
// TS 默认只认 ES 模块。但用了 Webpack 之类的构建工具，是支持以模块形式导入非 ES 模块的，比如导入了一个 CSS：
//import 'normalize.css';这样 TS 不识别，会报错，所以要先把它们声明出来。
declare module 'vue3-table-lite'

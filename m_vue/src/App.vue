<template>
    <div id="nav">
    <router-link to="/">Home</router-link> |
    <router-link to="/about">About</router-link> |
    <router-link to="/tail">Tail</router-link> |
    <router-link to="/stock">Stock</router-link> |
    <router-link to="/test">Test</router-link>
    <router-view/>
    </div>
</template>

<style lang="scss">
// font-family最后加上sans-serif，也是为了保证能够调用这个字体族里面的字体，因为大多数计算机里都有这种字体
// -webkit-font-smoothing有三个属性 none ------ 对低像素的文本比较好 subpixel-antialiased ------默认值 antialiased ------抗锯齿很好
// Gecko内核的浏览器火狐,针对ios系统制定了一个渲染规则-moz-osx-font-smoothing: inherit | grayscale 设置grayscale对于图标字体表现更清晰。
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {
  padding: 30px;

  a {
    font-weight: bold;
    color: #2c3e50;

    &.router-link-exact-active {
      color: #42b983;
    }
  }
}
</style>

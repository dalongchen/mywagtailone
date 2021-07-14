<template>
    <div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <HelloWorld msg="Welcome to Your Vue.js + TypeScript App"/>
    </div>
    <!--文件在当前目录./  Page2.html访问Page2Image.jpg-->
    <!--<img src=”./Page2Image.jpg”>或者<img src=”Page2Image. jpg”>-->
</template>

<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import HelloWorld from '@/components/HelloWorld.vue'; // @ is an alias to /src
@Options({
  components: {
    HelloWorld,
  },
})
export default class Home extends Vue {}
</script>

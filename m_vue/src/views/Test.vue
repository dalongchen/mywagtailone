<template>
  <el-container>
    <el-header style="height: 160px; border: 1px solid #eee">
      <el-row  class="row-bg" >
        <el-col :span="1.1">
          <div id="counter">
            Counter: {{ counter }}
          </div>
        </el-col>
      </el-row>
    </el-header>
    <el-main>
    </el-main>
  </el-container>
</template>
<script lang="ts"> 
import { defineComponent } from "vue";
// import axios from "axios";
// import { defineComponent, ref } from "vue";
// import TestSon from "@/components/TestSon.vue";

export default defineComponent({
  name: "Test",
  data() {
    return {
      counter: 0,
    }
  },
  mounted() {
    setInterval(() => {
      this.counter++
    }, 1000)
  },
  methods: {
  },
});
</script>
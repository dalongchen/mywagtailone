<template>
  <el-container>
    <el-header style="height: 160px; border: 1px solid #eee">
      <el-row  class="row-bg" >
        <el-col :span="1.1">
            <el-button round @click="choice_stock('research_report')" :disabled="f">
              {{research_report}}
            </el-button>
        </el-col>
        <el-col :span="1.05">
          <el-link :disabled="false">默认链接</el-link>
          <el-link type="primary" :disabled="true">主要链接</el-link>
        </el-col>
      </el-row>
    </el-header>
    <el-main>
    </el-main>
  </el-container>
</template>
<script lang="ts"> 
import { defineComponent } from "vue";
import axios from "axios";
// import { defineComponent, ref } from "vue";
// import TestSon from "@/components/TestSon.vue";

export default defineComponent({
  name: "Test",
  data() {
    return {
      research_report:"研告和机调数量(1年)",
      f:false,
    }
  },
   methods: {
    choice_stock(d:string) {
      this.research_report = "工作中约30秒";
      this.f = true;
      axios.get("http://127.0.0.1:8000/datatables/artificial_intelligence/", {
        params: { s: d}
      }).then(response => {
        this.f = false;
        let r =response.data.number
        // console.log(response.data.number);
        if (d == "research_report"){
          this.research_report = r;
        }
      }).catch(error => {
        this.f = false;
        console.log(error);
      });
    },
  },
});
</script>